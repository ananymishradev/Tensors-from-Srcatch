# Tensor Internals: Detailed Notes (Sections 1-17)

---

## 1. PyTorch Source Files to Read for Tensor Implementation

I am reading the PyTorch files in this order while building the from-scratch Tensor in this repository. Paths are relative to a clone of [pytorch/pytorch](https://github.com/pytorch/pytorch).


### 1A. Storage and Tensor metadata (build Sections 2-6 first)

| File or directory | What to learn for this repo |
| --- | --- |
| `c10/core/TensorImpl.h` | The authoritative definition of `sizes`, `strides`, `storage_offset`, `data_type`, `device`, `layout`. This is the struct my `Tensor` metadata mirrors. Note how sizes/strides are stored as `c10::IntArrayRef` and how `storage_offset` is a single integer. |
| `c10/core/StorageImpl.h` | The flat resizable byte buffer behind every tensor: `data_ptr`, `nbytes`, refcounting, allocator reference, device tag. This is the model for my `Storage` class. |
| `c10/core/Device.h`, `c10/core/ScalarType.h`, `c10/core/Layout.h`, `c10/core/MemoryFormat.h` | The exact enums for `cpu/cuda/mps/meta`, `float32/float16/bfloat16/int64/bool`, `strided/sparse`, and `contiguous/channels_last`. Copy these concepts as simple Python enums first. |
| `aten/src/ATen/core/Tensor.h` | How the C++ `Tensor` handle wraps `TensorImpl` with reference counting. Explains why views in this README share storage: they share the same `StorageImpl` with different `TensorImpl` metadata. |
| `torch/csrc/autograd/python_variable.cpp` | How the Python `torch.Tensor` object binds to C++ `TensorImpl`. Useful to see which Python attributes (`size()`, `stride()`, `storage_offset()`, `data_ptr()`) are thin wrappers over `TensorImpl` fields. |

Start here in code: implement `Storage(nbytes, device)`, `Tensor(sizes, strides, storage_offset, dtype, device)`, element-offset addressing from Section 3, and the `data_ptr` versus storage-identity distinction from Section 5.

### 1B. Views, shapes, and copies (build view semantics next)

| File or directory | What to learn for this repo |
| --- | --- |
| `aten/src/ATen/native/TensorShape.cpp` | Reference implementations of `view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `narrow`, `expand`, `as_strided`, and contiguity checks. Read this to get the exact compatibility rules for `view` and the zero-stride trick for `expand`. |
| `aten/src/ATen/native/Resize.cpp` and `aten/src/ATen/native/Repeat.cpp` | How resize, repeat, and tiling allocate new storage versus returning views. Compare with `clone` and `contiguous` behavior described in Section 4. |
| `aten/src/ATen/MemoryOverlap.cpp` (search `assert_no_internal_overlap` / overlap checks) | How PyTorch detects dangerous memory overlap for in-place ops. Relevant before implementing in-place ops and version counters in Section 8. |

Start here in code: implement `transpose`, slicing with offset/size/stride adjustment, `expand` with stride `0`, `view` with compatibility check, `reshape` with view-or-copy fallback, and `clone` / `contiguous`.

### 1C. Dispatcher and CPU kernels (build the operator layer next)

| File or directory | What to learn for this repo |
| --- | --- |
| `aten/src/ATen/core/dispatch/` and `c10/core/DispatchKey.h` | How an operator name plus dispatch keys `(device, dtype, layout, autograd, vmap, autocast)` select a kernel. My from-scratch dispatcher only needs `(dtype, device)` keys, but the registration pattern is the same. |
| `aten/src/ATen/native/cpu/` | Small readable CPU kernels for `add`, `mul`, reductions, and `relu`. Use these as pseudocode for the Python kernels registered in my dispatcher. |
| `aten/src/ATen/native/cuda/` (skim only) | How the same op name maps to a different kernel file per device, and where channels-last checks appear. Do not port CUDA code; just mirror the per-device registration idea. |

Start here in code: implement `register_kernel(op_name, device, dtype, fn)` plus `dispatch(op_name, *tensors)` covering `add`, `mul`, `matmul`, `sum`, and `relu`.

### 1D. Autograd (build backward tracking last)

| File or directory | What to learn for this repo |
| --- | --- |
| `torch/csrc/autograd/variable.h` | How autograd metadata (`requires_grad`, `grad_fn`, `grad`, version counter) attaches to a tensor without changing its storage layout. This maps directly to the autograd fields on my `Tensor`. |
| `torch/csrc/autograd/function.h` and `torch/csrc/autograd/engine.cpp` | How `Function` nodes record `save_for_backward` tensors and how `engine.execute` walks the graph in reverse topological order. This is the algorithm for my `backward()` implementation. |
| `torch/csrc/autograd/python_function.cpp` (search `save_for_backward`) | Exact semantics of which tensors an op must save and why saved tensors dominate memory, as discussed in Section 8. |

Start here in code: implement `Function` with `forward` / `backward`, `grad_fn` links, `save_for_backward`, topological `backward()`, `no_grad` guard, and the version-counter error for in-place modification of saved tensors.

### 1E. Memory and compilation (read after the core Tensor works)

| File or directory | What to learn for this repo |
| --- | --- |
| `c10/cuda/CUDACachingAllocator.cpp` (read the header comments first) | Block pooling, splitting, and why `empty_cache` rarely helps. Informs the simple free-list or pool I will use to model caching behavior from Section 9. |
| `torch/_dynamo/` and `torch/_inductor/` (Python directories) | Guards, FX graph capture, and pointwise/reduction fusion behind `torch.compile`. Read at a high level only; replicate just the idea of fusing a `sin + cos + sum` sequence in my own tests. |
| `torch/csrc/autograd/python_anomaly_mode.cpp` (optional) | How anomaly detection tracks which forward op produced a NaN. Useful inspiration for debug hooks in my implementation. |

Practical tip: read headers (`.h`) before implementation files (`.cpp`). Headers state the data model in a few hundred lines; `.cpp` files contain the full edge-case handling. For every file above, write one small test in this repo that asserts my implementation matches the documented PyTorch behavior before moving on.

---

## 2. Core Model: Tensor as View over Storage

A `torch.Tensor` seen from Python is a small descriptor object. The numerical data lives in a separate flat memory allocation. Conceptually:

```text
torch.Tensor (Python object)
  -> TensorImpl (C++ object holding metadata)
       sizes:           list of dimension lengths, e.g. [2, 3]
       strides:         step in elements for each dimension, e.g. [3, 1]
       storage_offset:  starting element index inside storage, e.g. 0
       dtype:           element type, e.g. float32
       device:          cpu, cuda:0, mps, meta, etc.
       layout:          strided, sparse_coo, sparse_csr, etc.
       autograd meta:   requires_grad, grad_fn, saved version counter
  -> UntypedStorage (flat, refcounted byte buffer on a specific device)
  -> grad Tensor (if requires_grad; has its own separate TensorImpl + Storage)
```

Three consequences follow from this split:

1. Multiple tensors can share one storage. Transpose, slicing, `view`, `expand`, and `narrow` normally do not copy data.
2. Tensor metadata operations are O(1). Changing shape or strides only rewrites a few integers in `TensorImpl`.
3. Lifetime is tied to storage, not to the logical view. A small slice keeps the entire parent storage alive until all views are freed.

The `.grad` field is a full tensor with its own storage, not a view into the forward storage. Saving activations for backward is therefore a separate allocation cost from parameters and gradients.

## 3. Sizes, Strides, and Storage Offset

For a contiguous strided tensor, the location of element `index = (i0, i1, ..., ik)` is:

```text
element_offset = storage_offset + sum(ij * stride[j])
byte_address   = storage_base_address + element_offset * element_size(dtype)
```

Example for shape `[2, 3]` with row-major strides `[3, 1]`:

- Element `[0, 0]` is at offset `0`.
- Element `[0, 2]` is at offset `2`.
- Element `[1, 0]` is at offset `3`.
- Element `[1, 2]` is at offset `5`.

The same formula is used by NumPy. PyTorch and NumPy differ mainly in device support, autograd metadata, dispatch, and layout options, not in the addressing math.

```python
import torch

a = torch.zeros(2, 3)
print(a.size())            # torch.Size([2, 3])
print(a.stride())          # (3, 1) for a fresh contiguous tensor
print(a.storage_offset())  # 0
print(a.untyped_storage().data_ptr())  # identity of the underlying allocation
```

`stride()` is reported in elements, not bytes. To convert to bytes, multiply by `tensor.element_size()`.

## 4. Views, Copies, and Lifetime Pitfalls

### 4.1 Operations that return views

These operations share storage with the input and run in O(1) metadata time:

- `transpose`, `permute`, `t()`
- Basic slicing: `a[1, :]`, `a[0:10, 0:10]`, `a[::2]`
- `view`, when the requested shape is compatible with the current strides
- `expand` / broadcasting
- `narrow`, `select`, `unsqueeze`, `squeeze` (squeeze/unsqueeze only change metadata)

```python
import torch

a = torch.zeros(2, 3)
t = a.transpose(0, 1)

print(t.size())    # torch.Size([3, 2])
print(t.stride())  # (1, 3): strides were swapped, data was not moved
print(t.untyped_storage().data_ptr() == a.untyped_storage().data_ptr())  # True: shared

v = a[1, :]
print(v.storage_offset())  # 3: the view starts 3 elements into the same storage
```

A broadcasted tensor uses stride `0` in the broadcast dimension. No data is duplicated; the same element is revisited logically.

### 4.2 Operations that copy

These operations allocate new storage:

- `clone`, `contiguous` (when already contiguous, `contiguous()` may return self without copying; otherwise it copies)
- `to(dtype=...)` or `to(device=...)` when a conversion is actually needed
- Advanced indexing: boolean masks, integer array indexing, `torch.index_select` with arbitrary indices
- `astype`-equivalent dtype casts, `torch.cat` along a new allocation, most reductions that return a smaller tensor

Rule of thumb: basic slices and stride manipulations are views; fancy indexing, dtype conversion, device transfer, and explicit `clone`/`contiguous` are copies. When in doubt, compare `untyped_storage().data_ptr()` or check `tensor._base` / NumPy `base`.

### 4.3 Lifetime pitfall: small view, large storage

Because a view holds a reference to its storage, slicing 10 elements out of a 10 GB tensor keeps all 10 GB alive as long as the slice exists.

```python
import torch

t = torch.zeros(1000, 1000)  # ~4 MB for float32
s = t[0:10, 0:10]

print(t.untyped_storage().nbytes())
print(s.untyped_storage().nbytes())  # same size: s shares t's allocation
print(s.clone().untyped_storage().nbytes())  # small: clone materializes only the view
```

Use `.clone()` after slicing when the parent tensor can be freed and only a small region is needed long-term.

### 4.4 `view` versus `reshape` versus `contiguous`

- `view` never copies. It succeeds only if the new shape can be expressed with the existing strides and offset. Otherwise it raises.
- `reshape` tries to return a view; if that is impossible, it silently copies and returns a contiguous tensor.
- `contiguous` returns a row-major (C-order) tensor with the same values. If the input is already contiguous in the requested memory format, it returns the same tensor without copying.

Before calling `.view()` on a tensor that may be transposed, sliced with a step, or produced by `expand`, call `.contiguous()` first or use `.reshape()` if a copy is acceptable.

## 5. Identity: `data_ptr`, Storage Identity, and `set_`

Two pointers are commonly confused:

- `tensor.data_ptr()` is the address of element `[0, 0, ..., 0]` of that logical tensor. It incorporates `storage_offset`.
- `tensor.untyped_storage().data_ptr()` is the base address of the shared allocation.

Two views of the same allocation therefore have equal storage pointers but different `data_ptr()` values when their offsets differ. To test whether two tensors share an allocation, compare storage pointers, not `data_ptr()`.

```python
import torch

a = torch.zeros(2, 3)
row1 = a[1, :]

print(a.data_ptr() == row1.data_ptr())  # False: different logical start
print(a.untyped_storage().data_ptr() == row1.untyped_storage().data_ptr())  # True: shared
```

`Tensor.set_(storage, storage_offset, size, stride)` is a low-level operation that replaces a tensor's metadata to point at an existing storage with explicit offset, size, and stride. It is used by library internals and advanced extension code. Normal application code should prefer `as_strided`, slicing, `view`, or `narrow`, because an incorrect `set_` call can create out-of-bounds views.

## 6. Dtypes, Devices, and Layouts

### 6.1 Dtypes and storage

Current PyTorch separates untyped memory from typed interpretation:

- `UntypedStorage` is a flat byte buffer.
- `TypedStorage` is a deprecated typed wrapper around it.
- `Tensor(dtype=...)` records how those bytes should be interpreted.

Common dtypes: `float32`, `float16`, `bfloat16`, `float64`, `int8`, `int16`, `int32`, `int64`, `bool`, `complex64`, `complex128`, and quantized types such as `quint8`.

`float16` and `bfloat16` both use 2 bytes per element but have different range/precision trade-offs. `bfloat16` preserves approximately the dynamic range of `float32` with reduced precision and is generally more stable for training than `float16` on hardware that supports it.

### 6.2 Devices

Storage is bound to one device: `cpu`, `cuda`, `mps`, `xpu`, `meta`, or `fake`. A CPU tensor and a CUDA tensor never share storage. Transfers with `.to(device)` or `.cuda()` allocate on the destination and copy.

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.zeros(2, 3, device=device)
print(a.device)
```

Use `torch.cuda.is_available()` guards in portable examples. CUDA-only APIs such as `torch.cuda.memory_summary()` raise or return empty results on CPU-only builds.

### 6.3 Layouts

- `torch.strided` (default, also called contiguous or NCHW for 4D image tensors).
- `torch.channels_last` (also called NHWC for 4D tensors): stores channel data together so that CUDA convolution kernels access memory coalescently. This can approximately double convolution throughput on modern NVIDIA GPUs when the model and hardware support it. It changes layout, not numerical results.
- `torch.sparse_coo`, `torch.sparse_csr`, `torch.sparse_csc`: same `TensorImpl` shape/stride prefix plus extra index/value arrays for sparse data.
- Quantized tensors add scale and zero-point metadata.
- Meta tensors (`device="meta"`) have shape, stride, dtype, and device but no data buffer. Fake tensors (`FakeTensorMode`) add shape-only execution for `torch.compile` and export. Both let PyTorch run the dispatcher and infer output shapes without allocating real memory.

## 7. The Dispatcher

Every PyTorch operation goes through a central dispatcher. Conceptually, the dispatcher is a table keyed by operator name plus a set of dispatch keys:

```text
(operator, dtype, device, layout, autograd state, vmap, autocast, functionalization, tracing)
```

For `c = a + b`, the path is approximately:

1. Python calls `Tensor.__add__`, which enters the C++ dispatcher as `aten::add.Tensor`.
2. The dispatcher computes the dispatch key set from the inputs and thread-local state.
3. If any input requires grad and grad mode is enabled, the `Autograd` key routes first to a generated autograd wrapper. That wrapper saves what backward will need and creates a `grad_fn` node.
4. The dispatcher re-dispatches with autograd excluded (`AutoNonVariableTypeMode`) to the concrete backend kernel: a CPU kernel, a CUDA kernel, an MPS kernel, or a vendor library such as cuBLAS, cuDNN, NCCL, or a fused FlashAttention kernel.

Practical implications:

- A custom operator registered once through the dispatcher infrastructure automatically composes with autograd, `vmap`, autocast, and `torch.compile` if its decomposition or kernel registrations cover those keys.
- `torch.no_grad()` and `torch.inference_mode()` remove autograd keys from the dispatch set. This skips graph construction entirely, which is faster than merely not calling `.backward()`.
- `torch.inference_mode()` is stricter and faster than `torch.no_grad()` for pure inference because it also forbids creating tensors that require grad and allows additional optimizations. Do not use it where backward will be needed later.

## 8. Autograd: Define-by-Run Graph of Functions

Autograd builds a directed acyclic graph incrementally as code runs. This is called define-by-run: Python control flow (`if`, `for`, `while`) directly changes the graph on each forward pass, unlike older static-graph frameworks where the graph had to be declared up front.

Each differentiable op creates a `Node` (a `torch.autograd.Function`) linked through `grad_fn`:

```python
import torch

x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()

print(y.grad_fn)  # e.g. <SumBackward0 ...>: records how to differentiate the sum
y.backward()
print(x.grad)  # d/dx sum(x^2) = 2*x
```

Calling `loss.backward()` traverses that graph in reverse topological order and accumulates gradients into `.grad` of leaf tensors with `requires_grad=True`.

Three details dominate real training memory and correctness:

1. **Saved tensors.** Ops keep inputs or outputs alive for backward via `save_for_backward` (visible internally as `_saved_*` attributes). In transformers, activations often use 4-8x more memory than parameters, which is why activation checkpointing (`torch.utils.checkpoint`) trades recomputation for memory.
2. **In-place operations and version counters.** Every tensor has a version counter incremented by in-place ops (`add_`, `relu_`, `copy_`, `+=`). If an op saved a tensor for backward and that tensor is modified in place before backward runs, autograd raises a version-mismatch error. Avoid in-place ops on tensors needed for backward unless memory pressure requires them.
3. **Hooks.** `tensor.register_hook(fn)`, `tensor.register_post_accumulate_grad_hook(fn)`, and `torch.autograd.graph.register_multi_grad_hook` / pack hooks allow gradient clipping, offloading, compression, logging, and distributed synchronization. Pack hooks run with grad disabled; returned packed values should be detached or plain data to avoid reference cycles.

Minimal no-grad example:

```python
import torch

x = torch.randn(3, requires_grad=True)

with torch.no_grad():
    z = x * 2

print(z.requires_grad)  # False: constructed without tracking
print(z.grad_fn)        # None
```

## 9. Memory Management

### 9.1 CPU versus CUDA allocation

CPU tensors use the process allocator directly. CUDA tensors use PyTorch's caching allocator: freed CUDA blocks are retained in a per-device pool and reused instead of being returned immediately to the driver. This avoids expensive `cudaMalloc`/`cudaFree` synchronization but means `nvidia-smi` may show high reserved memory even after tensors are deleted.

Consequences:

- `torch.cuda.empty_cache()` releases unused cached blocks back to the driver. It does not free memory still referenced by tensors, does not defragment, and usually does not speed up training. Use it for debugging or before launching a new job phase, not in the training loop.
- `torch.cuda.memory_summary()` and `torch.cuda.memory_snapshot()` are the primary tools for distinguishing allocated memory, reserved cache, active blocks, and fragmentation.
- The allocator splits large blocks on demand. Repeated allocation of many differently sized short-lived tensors fragments the pool. Reusing buffers, fixing batch shapes, and enabling `torch.compile` reduce fragmentation.

### 9.2 Reducing training memory

Common techniques, in approximate order of adoption:

1. **Mixed precision with `torch.autocast`.** Computing in `float16` or `bfloat16` approximately halves activation bytes and speeds up matmuls on Tensor Cores. Pair with `torch.amp.GradScaler` for `float16`; `bfloat16` usually does not need scaling.
2. **Gradient accumulation.** Run several micro-batches, accumulate `.grad`, and step once to simulate a larger batch without allocating the full batch at once. Remember to scale the loss correctly and to zero grads at the right cadence.
3. **Activation checkpointing.** Discard selected activations in forward and recompute them in backward (`torch.utils.checkpoint.checkpoint`). Saves memory at the cost of extra compute.
4. **Optimizer state sharding and offload.** For large models, use sharded optimizers (FSDP/DeepSpeed-style) or CPU offload rather than holding full optimizer state on every GPU.

## 10. Contiguity and Channels-Last Layout

Call `.contiguous()` (or `.contiguous(memory_format=torch.channels_last)`) in these cases:

- Before `.view()` when the tensor may be non-contiguous (transposed, strided slice, expanded).
- Before custom CUDA/C++ extensions that assume dense row-major input.
- Before a `torch.compile` region where alternating strided views cause guard failures and recompiles.

Do not call `.contiguous()` unconditionally in hot paths. Each effective call is an O(n) copy. Check `tensor.is_contiguous()` or `tensor.is_contiguous(memory_format=torch.channels_last)` when the layout is uncertain.

```python
import torch

c = torch.randn(4, 4)
print(c.is_contiguous())      # True
print(c.T.is_contiguous())    # False: transpose only swapped strides

d = c.T.contiguous()
print(d.is_contiguous())      # True after an explicit copy
print(d.untyped_storage().data_ptr() == c.untyped_storage().data_ptr())  # False
```

For image models on CUDA, consider channels-last end to end:

```python
import torch

model = torch.nn.Conv2d(3, 64, 3).cuda().to(memory_format=torch.channels_last)
x = torch.randn(8, 3, 224, 224, device="cuda").to(memory_format=torch.channels_last)
y = model(x)
print(y.is_contiguous(memory_format=torch.channels_last))
```

Convert model and input once at the boundary, not per layer.

## 11. `torch.compile` (Dynamo + Inductor)

`torch.compile` is a JIT stack composed of:

- **TorchDynamo:** traces Python bytecode at frame boundaries into an FX graph, specializing on shapes, dtypes, devices, and Python state (guards).
- **TorchInductor:** lowers the FX graph, fuses pointwise operations and reductions, chooses layouts, and emits Triton kernels for GPU or C++ for CPU.

Typical steady-state speedups over eager mode are in the range of 1.3x to 2x for compute-bound models, with larger gains when many small ops fuse. The first call is slow because it compiles. Later calls are fast as long as guards hold.

```python
import torch

def fn(t):
    return (t.sin() + t.cos()).sum()

compiled_fn = torch.compile(fn)
x = torch.randn(1024, 1024)
print(compiled_fn(x))
```

Operational notes:

- Keep shapes stable. Dynamic batch/sequence lengths without `dynamic=True` or `torch._dynamo.mark_dynamic` create a new specialization and recompile per shape.
- Set `TORCH_LOGS="recompiles"` or use `torch._dynamo.explain` to find why a function recompiles.
- Per-op Python stack traces and some in-place mutation patterns are harder to debug under compilation. Get the eager version correct first, then compile.
- `mode="reduce-overhead"` (CUDA graphs) helps small-batch launch overhead; `mode="max-autotune"` spends more compile time searching for faster kernels.

## 12. Writing C++/CUDA Extensions: TensorAccessor

Raw `data_ptr<float>()` plus manual index math is error-prone because it ignores strides. `TensorAccessor` preserves stride handling with minimal overhead:

```cpp
// checked once at construction, unchecked indexing afterwards
auto x = input.accessor<float, 3>();  // 3-dimensional float tensor
float v = x[i][j][k];
```

- Use `accessor<T, N>()` on CPU.
- Use `packed_accessor32<T, N>()` or `packed_accessor64<T, N>()` on CUDA. The 32-bit variant uses 32-bit integer indexing and is faster when tensor sizes fit in 32 bits.
- Accessors assume the tensor will not be resized during the kernel. Ensure contiguity or handle arbitrary strides explicitly if the kernel requires it.

## 13. Which Container to Use When

| Task | Preferred container | Notes and pitfalls |
| --- | --- | --- |
| Heterogeneous small data, arbitrary Python objects | Python `list`, `dict`, `set` | Flexible but boxed. Avoid `x in list` (O(n)), `list.insert(0, ...)` (O(n)), and string concatenation with `+` in a loop (use `join`). |
| Homogeneous numeric arrays, vectorized math on CPU | NumPy `ndarray` | Packed C buffers plus ufuncs. Avoid Python-level loops, repeated `np.concatenate` in a loop (preallocate or collect then concatenate once), and inner-loop access along a non-contiguous axis. |
| Labeled data, time series, group-by/join, missing data, CSV/Parquet IO | pandas `DataFrame` / `Series` | Index alignment and IO conveniences. Avoid row-wise `iterrows` (use vectorized ops, `itertuples`, or Numba), incremental column insertion that fragments the BlockManager, and `object` dtype for strings when `string[pyarrow]` or `category` fits. |
| GPU compute, autograd, neural networks | `torch.Tensor` | Dispatcher plus fused kernels. Avoid `.view()` on non-contiguous tensors without `.contiguous()`, retaining slices of very large storages, and fine-grained per-op Python overhead in hot loops (use `torch.compile`). |
| Text-heavy columns and Parquet interchange | pandas with PyArrow backend (`string[pyarrow]`, `dtype_backend="pyarrow"`) | Consistent `NA` semantics, fast IO, and Arrow zero-copy to Polars/PyArrow. Avoid mixing NumPy `NaN` semantics with Arrow `NA` in the same pipeline. |
| C-speed hot loop over NumPy data that cannot be vectorized | Cython memoryviews or Numba | Requires contiguous input (`::1` in Cython) for best results. See the extension sections of a Cython/Numba guide for setup. |

## 14. Zero-Copy Bridges Between Libraries

Prefer shared-memory handoffs and verify sharing instead of assuming it. Useful checks are NumPy `.base`, `tensor.untyped_storage().data_ptr()`, and Arrow `zero_copy_only=True` imports where available.

- `list -> np.array`: copies (boxes Python objects into a packed buffer).
- `NumPy <-> pandas`: `df.to_numpy()` and `pd.DataFrame(arr)` are often views for single-dtype data without missing-value conversion; mixed dtypes, extension arrays, or alignment force a copy.
- `NumPy <-> PyTorch` on CPU: `torch.from_numpy(arr)` and `tensor.numpy()` share memory when dtypes match. The NumPy array must be writable and CPU-resident; some dtypes have no exact counterpart.
- `NumPy / PyTorch <-> Arrow`: DLPack (`torch.to_dlpack` / `torch.from_dlpack`), `__cuda_array_interface__` for GPU exchange, and `pa.Tensor.from_numpy` for CPU. Device, dtype, and contiguity constraints apply.
- `bytes` / `bytearray` / `mmap <-> NumPy / PyTorch`: the buffer protocol (`np.frombuffer`, `torch.frombuffer`) avoids a copy for read-compatible paths. Writable access requires a mutable buffer such as `bytearray` or `mmap`.

## 15. Memory Footprint Reference

Approximate sizes for 1,000,000 elements, excluding Python object overhead and allocator rounding:

| Representation | Bytes |
| --- | --- |
| Python `list[int]` | ~28-36 MB plus pointer array, scattered across many allocations |
| NumPy `int64` | 8 MB packed |
| NumPy `int32` | 4 MB packed |
| NumPy / Torch `float16` / `bfloat16` | 2 MB packed |
| Torch `float32` on CPU, no grad | 4 MB |
| Torch `float32` with grad | ~8 MB for data plus grad, plus autograd graph and saved activations |
| pandas `object` column of strings | Similar to a Python list plus index overhead |
| pandas `category` with low cardinality | Often 1-2 MB for the codes plus a small dictionary |

Measure real jobs with `sys.getsizeof` (shallow), `ndarray.nbytes`, `tensor.nbytes` / `untyped_storage().nbytes()`, `df.memory_usage(deep=True)`, `tracemalloc` or `memray` for peak Python RAM, and `torch.cuda.memory_summary()` for GPU.

## 16. Hands-On Labs

General method for every lab: time-box the work, record machine, OS, Python, NumPy, pandas, and PyTorch versions, and write down the prediction before running the code. Labs marked Beginner assume only basic Python; Intermediate assumes comfort with NumPy/pandas; Advanced assumes comfort with PyTorch training loops.

### L1 (Beginner, 15 min): Feel Python boxing cost

Goal: quantify why a packed NumPy buffer beats a Python list for numeric work.

```python
import sys
import numpy as np

l = list(range(100_000))
print(sys.getsizeof(l))     # size of the pointer array, not the ints
print(sys.getsizeof(l[0]))  # size of one boxed int object

a = np.arange(100_000, dtype=np.int64)
print(a.nbytes, a.itemsize)  # total packed bytes and bytes per element
```

In IPython, also run:

```python
%timeit sum(l)
%timeit a.sum()
```

Done when: you can explain the observed speed gap in terms of pointer chasing plus per-object boxing versus one contiguous C loop, and state both byte counts.

### L2 (Beginner, 30 min): Strides playground

Goal: predict strides and explain why transpose, reshape (when compatible), slicing, and broadcasting are O(1).

```python
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

x = np.arange(12).reshape(3, 4)
print(x.strides, x.flags)
print(x.T.strides, x.T.base is x)
print(x[::-1].strides, x[::2, ::3].strides)
print(np.broadcast_to(np.arange(4), (3, 4)).strides)  # expect a 0 stride
print(sliding_window_view(np.arange(10), 3).strides)
```

Done when: you predicted each `strides` tuple before running, and can explain negative strides, stepped-slice strides, and 0-stride broadcasting.

### L3 (Beginner, 30 min): Views versus copies trap

Goal: state the view/copy rule from evidence and fix the large-storage retention bug.

```python
import numpy as np
import torch

x = np.arange(6, dtype=np.int32)
y = x[2:]
y[0] = 99
print(x)  # x changed: y is a view

z = x.copy()
z[0] = -1
print(x[0] != -1)  # True: copy is independent

t = torch.zeros(1000, 1000)
s = t[0:10, 0:10]
print(t.untyped_storage().nbytes(), s.untyped_storage().nbytes())  # equal: shared
print(s.clone().untyped_storage().nbytes())  # small after clone
```

Done when: you can state the rule (basic slice/transpose/broadcast return views; fancy indexing, dtype casts, and explicit `clone`/`contiguous` copy) and show the `clone()` fix for a small slice of a large tensor.

### L4 (Intermediate, 45 min): dict and set mechanics

Goal: connect hashing, load factor, and deletion handling to observed behavior.

```python
print(hash(-1), hash(7))
d = {}
for i in range(10000):
    d[i] = i

import sys
print(sys.getsizeof(d))

s = {1, 2, 3}
s.discard(2)
print(2 in s)
```

Also try `PYTHONHASHSEED=0 python -c 'print(hash("hello"))'` versus two runs without the fixed seed to see string hash randomization (SipHash-based protection against hash-flooding denial of service).

Done when: you can explain why `-1` hashes to `-2`, why string hashes differ across runs by default, what load factor (~2/3) has to do with table growth, how dummy/tombstone entries keep lookups correct after deletion, and how the compact table preserves insertion order.

### L5 (Intermediate, 1 hour): pandas Blocks and fragmentation

Goal: diagnose BlockManager fragmentation and fix it with a single construction.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"a": np.arange(5), "b": np.arange(5.0)})
print(df._mgr)
print([str(b.dtype) for b in df._mgr.blocks])

df2 = pd.DataFrame({"x": [1]})
for i in range(105):
    df2[i] = i  # repeated insertion: watch for PerformanceWarning

print(len(df2._mgr.blocks))
print(df2._mgr.is_consolidated())

fixed = pd.concat([pd.DataFrame({i: [i]}) for i in range(10)], axis=1)
print(fixed.shape)
print(pd.Series(["a", "b"] * 5000, dtype="category").memory_usage(deep=True))
```

Done when: you can show a fragmented frame, replace incremental insertion with one `pd.concat(...)` plus `.copy()` where needed, and demonstrate memory savings from `category` or `string[pyarrow]` for suitable columns.

### L6 (Intermediate, 1 hour): Cache and contiguity benchmark

Goal: link summation-axis performance to memory order and cache lines.

```python
import numpy as np

a = np.random.rand(2000, 2000)
print(a.flags)

# In IPython/Jupyter:
# %timeit a.sum(axis=0)
# %timeit a.sum(axis=1)
# %timeit a.T.sum(axis=1)
# print(a.T.flags)
# b = np.asfortranarray(a)
# %timeit b.sum(axis=0)

# Without IPython, use time.perf_counter around the same calls.
```

Done when: you can explain the winner from strides and cache locality, state the values of `C_CONTIGUOUS` / `F_CONTIGUOUS`, and predict how Fortran order flips the result.

### L7 (Intermediate, 1 hour): Arrow backend and Copy-on-Write

Goal: load with Arrow dtypes, show a Parquet round-trip, and demonstrate zero-copy exchange.

```python
import pandas as pd

pd.options.mode.copy_on_write = True
df = pd.read_csv("data.csv", engine="pyarrow", dtype_backend="pyarrow")
print(df.dtypes)
print(df.memory_usage(deep=True).sum())
print(df.convert_dtypes(dtype_backend="pyarrow").dtypes)
```

Extend this by writing `df.to_parquet("out.parquet")`, reading it back, and converting to `pyarrow.Table` or Polars without an intermediate NumPy copy where possible.

Done when: you show dtypes before/after conversion, a successful Parquet round-trip, and the exchange path used.

### L8 (Advanced, 1.5 hours): Torch storage and autograd memory

Goal: inspect saved tensors, version counters, no-grad dispatch, and compile behavior.

```python
import torch

x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
print(y.grad_fn)
y.backward()
print(x.grad)

if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
else:
    print("CUDA not available; ran on CPU.")

with torch.no_grad():
    z = x * 2
print(z.requires_grad)

compiled = torch.compile(lambda t: (t.sin() + t.cos()).sum())
print(compiled(torch.randn(8)))
```

To see the version-counter error intentionally, save `x` for backward in a custom `Function`, modify `x` in place before `backward()`, and observe the raised error. Then remove the in-place op.

Done when: you can explain which tensors backward retained, what `no_grad` changed in dispatch, and what speedup (if any) compile gave on your machine after warmup.

### L9 (Advanced, weekend): C-level reading

Goal: map one Python line to the C/C++ code that implements it.

Read `Objects/dictobject.c` (`lookdict`), `Objects/listobject.c` (`list_resize`), NumPy array constructors and stride logic, `pandas/core/internals/blocks.py` (`Block`), and `aten/src/ATen/native/cpu` kernels. Add temporary logging, rebuild the relevant project where feasible, rerun L2/L4 probes, and optionally collect `perf stat -e cache-misses` for the L6 benchmark.

Done when: you can trace one line such as `x.T` or `d[k]` through the C-level functions and stride/hash math involved.

### L10 (Advanced, 2 hours): End-to-end pipeline

Goal: build a CSV-to-training pipeline with no unnecessary copies.

Stages:

```text
CSV (pyarrow engine, Arrow dtypes)
  -> pandas cleaning (categorical/string dtypes, single concat, no fragmented inserts)
  -> to_numpy with explicit contiguous order
  -> torch.from_numpy (CPU share)
  -> .to(cuda, non_blocking=True), channels_last where appropriate
  -> torch.compile for the hot model region
  -> train
  -> back to pandas / Parquet for logging
```

Profile peak RAM (`memray`, `tracemalloc`) and wall time. At each handoff, record whether the transfer shared or copied by checking `.base`, `data_ptr`, storage pointers, and dtypes.

Done when: you deliver a one-page report listing each handoff, copy or share status, peak RAM, runtime, and the single largest remaining cost.

---

## 17. Common Pitfalls and Debugging Checklist

1. `.view()` fails with a contiguity error: insert `.contiguous()` or switch to `.reshape()` if a copy is acceptable.
2. GPU memory stays high after deleting tensors: check for lingering views, cached allocator reservation (`memory_summary`), and saved autograd graph references. `empty_cache()` only helps with unreferenced cache.
3. Small tensor keeps a huge allocation alive: `clone()` the slice and drop the parent.
4. In-place op error during `backward()`: remove in-place modification of tensors needed for gradient computation, or checkpoint/recompute them.
5. Unexpected copy on `torch.from_numpy` / `.numpy()`: verify CPU device, matching dtype, and writability.
6. `torch.compile` recompiles every step: stabilize shapes, use `dynamic=True` / `mark_dynamic`, and inspect `TORCH_LOGS="recompiles"`.
7. Channels-last does not help: verify both model and input are channels-last and that the ops used have channels-last kernels; check `is_contiguous(memory_format=torch.channels_last)`.
8. Mixed-precision instability: prefer `bfloat16` where supported; for `float16`, use gradient scaling and check for overflow in reductions and softmax.
