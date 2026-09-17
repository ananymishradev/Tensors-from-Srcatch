# Tensors in PyTorch: Storage, Views, Dispatcher, Autograd, and Memory

I am implementing a Tensor library from scratch in Python and C++, modeled directly on how PyTorch implements `torch.Tensor`.

My goal is not to wrap NumPy or PyTorch, but to rebuild the core ideas myself: a flat `Storage` buffer plus a `Tensor` view defined by `sizes`, `strides`, `storage_offset`, `dtype`, and `device`, with correct view-versus-copy semantics, broadcasting, autograd tracking, and memory behavior.

This repo is the design and study document for that implementation. Each section below defines one piece I will build: addressing math, slicing and transpose as O(1) views, `view` / `reshape` / `contiguous` rules, dtype and device handling, a small dispatcher for CPU kernels, define-by-run autograd with saved tensors and version counters, a caching-style memory model, and finally `compile`-style fusion concepts.

Planned implementation order:

1. `Storage` as a flat, refcounted 1D buffer with byte size, element size, and device tag.
2. `Tensor` metadata (`sizes`, `strides`, `storage_offset`, `dtype`) plus element-offset addressing.
3. View operations: `transpose`, `permute`, slicing, `narrow`, `squeeze` / `unsqueeze`, `expand` with zero strides.
4. Copy operations: `clone`, `contiguous`, dtype casts, fancy indexing.
5. `view` / `reshape` compatibility checks and channels-last style layouts.
6. Minimal CPU dispatcher and registered kernels (`add`, `mul`, `matmul`, `sum`, `relu`).
7. Define-by-run autograd (`Function`, `grad_fn`, `save_for_backward`, `backward`, `no_grad`).
8. Tests that verify sharing with storage identity, lifetime behavior, gradient correctness, and memory footprint.

## Contents

- [tensor-internals.md](./tensor-internals.md) — detailed notes, Sections 1-17
- [Resources](#resources) — reading list (in this file, below)

## Detailed Guide (Sections 1-17)

All in-depth material lives in [tensor-internals.md](./tensor-internals.md):

1. [PyTorch Source Files to Read for Tensor Implementation](./tensor-internals.md#1-pytorch-source-files-to-read-for-tensor-implementation)
2. [Core Model: Tensor as View over Storage](./tensor-internals.md#2-core-model-tensor-as-view-over-storage)
3. [Sizes, Strides, and Storage Offset](./tensor-internals.md#3-sizes-strides-and-storage-offset)
4. [Views, Copies, and Lifetime Pitfalls](./tensor-internals.md#4-views-copies-and-lifetime-pitfalls)
5. [Identity: `data_ptr`, Storage Identity, and `set_`](./tensor-internals.md#5-identity-dataptr-storage-identity-and-set_)
6. [Dtypes, Devices, and Layouts](./tensor-internals.md#6-dtypes-devices-and-layouts)
7. [The Dispatcher](./tensor-internals.md#7-the-dispatcher)
8. [Autograd: Define-by-Run Graph of Functions](./tensor-internals.md#8-autograd-define-by-run-graph-of-functions)
9. [Memory Management](./tensor-internals.md#9-memory-management)
10. [Contiguity and Channels-Last Layout](./tensor-internals.md#10-contiguity-and-channels-last-layout)
11. [`torch.compile` (Dynamo + Inductor)](./tensor-internals.md#11-torchcompile-dynamo--inductor)
12. [Writing C++/CUDA Extensions: TensorAccessor](./tensor-internals.md#12-writing-ccuda-extensions-tensoraccessor)
13. [Which Container to Use When](./tensor-internals.md#13-which-container-to-use-when)
14. [Zero-Copy Bridges Between Libraries](./tensor-internals.md#14-zero-copy-bridges-between-libraries)
15. [Memory Footprint Reference](./tensor-internals.md#15-memory-footprint-reference)
16. [Hands-On Labs](./tensor-internals.md#16-hands-on-labs)
17. [Common Pitfalls and Debugging Checklist](./tensor-internals.md#17-common-pitfalls-and-debugging-checklist)

---

## Resources

This is the curated reading list for the from-scratch implementation. Start with the official documentation for exact API behavior, then read the internals material to understand why PyTorch is designed this way, then study the source files and the small teaching frameworks that rebuild the same ideas.

### Official PyTorch documentation

- Tensor basics, creation, indexing, views, and `stride` / `storage_offset` semantics:
  `https://pytorch.org/docs/stable/tensors.html`
- Storage and `UntypedStorage` (`data_ptr`, `nbytes`, lifetime, `set_`):
  `https://pytorch.org/docs/stable/storage.html`
- Autograd mechanics (`grad_fn`, `save_for_backward`, version counters, hooks, `no_grad`, `inference_mode`):
  `https://pytorch.org/docs/stable/notes/autograd.html`
- Automatic mixed precision (`autocast`, `GradScaler`):
  `https://pytorch.org/docs/stable/amp.html`
- `torch.compile`, TorchDynamo, TorchInductor, dynamic shapes, and logging:
  `https://pytorch.org/docs/stable/torch.compiler.html`
- CUDA memory management (`memory_summary`, `memory_snapshot`, caching allocator behavior):
  `https://pytorch.org/docs/stable/notes/cuda.html`
- Extending PyTorch with custom ops and C++/CUDA extensions (`TensorAccessor`, dispatch registration):
  `https://pytorch.org/docs/stable/notes/extending.html`

### PyTorch internals writing

- Edward Z. Yang, "PyTorch Internals" (2019). The best single overview of `TensorImpl`, Storage, the dispatcher, and autograd generation:
  `https://blog.ezyang.com/2019/05/pytorch-internals/`
- PyTorch dispatcher and operator-registration deep dives, including the ZeroEntropy series on dispatch keys, boxed/unboxed calls, and backend kernels. Search "PyTorch dispatcher ZeroEntropy" for the current mirror.
- PyTorch developer discussions and design docs on version counters, saved tensors, `inference_mode`, channels-last, and the caching allocator. These are linked from the docs pages above and from issues in `pytorch/pytorch`.

### Source code to read while implementing

Clone `https://github.com/pytorch/pytorch` and focus on these paths:

- `aten/src/ATen/core/TensorImpl.h` and `StorageImpl.h`: the exact fields summarized in Sections 2-6 of tensor-internals.md.
- `aten/src/ATen/native/cpu/`: reference CPU kernels for elementwise ops and reductions.
- `torch/csrc/autograd/`: generated `Function` nodes, `save_for_backward`, version-counter checks, and hook handling.
- `aten/src/ATen/native/cuda/`: allocator interaction and channels-last kernel selection.
- `torch/_dynamo/` and `torch/_inductor/`: guards, graph capture, fusion, and code generation behind `torch.compile`.

For NumPy stride behavior used in the labs:

- NumPy `ndarray` internals and `as_strided` documentation:
  `https://numpy.org/doc/stable/reference/arrays.ndarray.html`

### From-scratch teaching implementations

These projects rebuild autograd and tensor machinery at small scale and are useful models for this repository:

- Andrej Karpathy, `micrograd`: minimal scalar autograd engine, ideal before writing tensor-level backward:
  `https://github.com/karpathy/micrograd`
- `minitorch`: tensor Storage plus operators, broadcasting, and autograd in Python, closest in spirit to this project:
  `https://github.com/minitorch/minitorch`
- `tinygrad`: small tensor runtime with lazy evaluation, fusion, and multiple backends:
  `https://github.com/tinygrad/tinygrad`
