# TinyGrad Pills

*A tl;dr first-principles guide to understanding TinyGrad, GPU architecture, and machine learning systems*

## Welcome

This collection of "pills" (inspired by Nix Pills) teaches you everything you need to know to understand and contribute to TinyGrad. Each pill is focused, practical, and builds on previous knowledge.

**Target audience**: Embedded software engineers, systems programmers with CS/Linux background, familiar with kernel drivers, ioctls, and basic ML concepts (CNNs, RNNs, backprop).

## Reading Order

### Foundation Pills (Start Here)
1. **[Pill 0: Introduction & Philosophy](00-introduction.md)** - What is TinyGrad and why it exists
2. **[Pill 1: Tensor & UOp Fundamentals](01-tensor-uop.md)** - The two core abstractions
3. **[Pill 2: The Compilation Pipeline](02-compilation-pipeline.md)** - How code becomes GPU kernels

### GPU & Hardware Pills
4. **[Pill 3: GPU Architecture Primer](03-gpu-architecture.md)** - How modern GPUs work
5. **[Pill 4: NVIDIA GPU Deep Dive](04-nvidia-gpu.md)** - CUDA, PTX, compute capability
6. **[Pill 5: Device Backends & Runtime](05-device-runtime.md)** - HCQ, memory, and device abstraction

### Compiler Pills
7. **[Pill 6: Schedule & Memory Management](06-schedule-memory.md)** - From graphs to executable kernels
8. **[Pill 7: Code Generation & Optimization](07-codegen-opt.md)** - BEAM search, tensor cores, optimization
9. **[Pill 8: Pattern Matching & Graph Rewriting](08-pattern-matching.md)** - The magic of UOp transformations

### Practical Pills
10. **[Pill 9: Jetson AGX Orin & NV Backend](09-jetson-orin.md)** - Understanding the dev kit and NV=1
11. **[Pill 10: Benchmarking & Performance](10-benchmarking.md)** - Measuring and understanding performance
12. **[Pill 11: Machine Learning Primer](11-ml-primer.md)** - Quick review of ML concepts in TinyGrad

### Advanced Pills
13. **[Pill 12: Advanced Topics](12-advanced-topics.md)** - Tensor cores, JIT, multi-device
14. **[Pill 13: Contributing to TinyGrad](13-contributing.md)** - How to make meaningful contributions

## Quick Reference

### Key Files by Function
- **User API**: `tinygrad/tensor.py`
- **IR**: `tinygrad/uop/ops.py`
- **Compiler**: `tinygrad/codegen/`
- **Scheduler**: `tinygrad/engine/schedule.py`
- **NVIDIA Backend**: `tinygrad/runtime/ops_nv.py`

### Essential Environment Variables
- `DEBUG=3` - Show fused kernels
- `DEBUG=4` - Show generated code
- `DEBUG=7` - Show assembly
- `NV=1` - Use NVIDIA userspace driver
- `BEAM=3` - Enable kernel optimization search
- `VIZ=1` - Visualize graphs

## Philosophy

TinyGrad prioritizes:
1. **Simplicity** - Every line must earn its keep
2. **Hackability** - The whole stack is visible and modifiable
3. **Performance** - But not at the cost of readability
4. **Education** - Code that teaches as you read it

## Contributing

Found an error or want to improve a pill? PRs welcome! Keep the tl;dr style - dense with information but still readable.

## License

Same as TinyGrad (MIT)
