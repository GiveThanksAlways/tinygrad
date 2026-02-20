# TinyGrad Pills

*A tl;dr first-principles guide to understanding TinyGrad, GPU architecture, and machine learning systems — built on Jetson AGX Orin 64GB*

## Welcome

This collection of "pills" (inspired by [Nix Pills](https://nixos.org/guides/nix-pills/)) teaches you everything you need to know to understand and contribute to TinyGrad. Each pill is focused, practical, and builds on previous knowledge.

**Target audience**: Embedded software engineers, systems programmers with CS/Linux background, familiar with kernel drivers, ioctls, and basic ML concepts (CNNs, RNNs, backprop).

**Hardware**: Jetson AGX Orin 64GB running NixOS with the NV userspace driver (`NV=1`).

## Reading Order

### Foundation Pills (Start Here)
0. **[Pill 00: Introduction & Philosophy](00-introduction.md)** — What TinyGrad is and why it exists
1. **[Pill 01: Tensor & UOp Fundamentals](01-tensor-uop.md)** — The two core abstractions
2. **[Pill 02: The Compilation Pipeline](02-compilation-pipeline.md)** — How Python becomes GPU kernels (8 stages)

### GPU & Hardware Pills
3. **[Pill 03: GPU Architecture Primer](03-gpu-architecture.md)** — SMs, warps, memory hierarchy, roofline model
4. **[Pill 04: NVIDIA GPU Deep Dive](04-nvidia-gpu.md)** — PTX, compute capability, QMD, CUBIN, tensor cores
5. **[Pill 05: Device Backends & HCQ](05-device-backends.md)** — Hardware Command Queue, timeline signals, MMIO

### Compiler & Engine Pills
6. **[Pill 06: Schedule & Memory Management](06-schedule-memory.md)** — AST → ExecItem, cache, TLSF memory planner
7. **[Pill 07: BEAM Search](07-beam-search.md)** — Empirical kernel optimization, action space, parallel timing
8. **[Pill 08: Pattern Matching & Graph Rewriting](08-pattern-matching.md)** — UPat, PatternMatcher, graph_rewrite

### Jetson AGX Orin Pills
9. **[Pill 09: Jetson NV Backend, Part 1](09-jetson-nv-backend-pt1.md)** — TegraIface, ioctls, memory allocation, channel setup
10. **[Pill 10: Jetson NV Backend, Part 2](10-jetson-nv-backend-pt2.md)** — QMD race fix, memmove, local mem, BEAM fork fix
11. **[Pill 11: The Matvec Heuristic](11-matvec-heuristic.md)** — The 7-line fix that gave us 7.6× speedup

### Performance & Practical Pills
12. **[Pill 12: Benchmarking & Performance](12-benchmarking.md)** — 36.71 tok/s, roofline analysis, profiling, what matters
13. **[Pill 13: Machine Learning Primer](13-ml-primer.md)** — Training loop, layers, autograd, transformers in TinyGrad
14. **[Pill 14: Contributing to TinyGrad](14-contributing.md)** — Repo structure, testing, debugging, code style

## Quick Reference

### Key Files by Function
| Area | File |
|------|------|
| User API | `tinygrad/tensor.py` |
| IR & Pattern Matching | `tinygrad/uop/ops.py` |
| Scheduler | `tinygrad/engine/schedule.py` |
| Memory Planner | `tinygrad/engine/memory.py` |
| BEAM Search | `tinygrad/codegen/opt/search.py` |
| Heuristic | `tinygrad/codegen/opt/heuristic.py` |
| NVIDIA Backend | `tinygrad/runtime/ops_nv.py` |
| HCQ Abstraction | `tinygrad/runtime/support/hcq.py` |
| PTX Renderer | `tinygrad/renderer/ptx.py` |
| NN Layers | `tinygrad/nn/__init__.py` |
| Optimizers | `tinygrad/nn/optim.py` |

### Essential Environment Variables
| Variable | What |
|----------|------|
| `NV=1` | Use NVIDIA userspace driver (required for Orin) |
| `DEBUG=1-7` | Verbosity: 1=timing, 3=opts, 4=PTX source, 7=disasm |
| `BEAM=N` | BEAM search width (0=heuristic only) |
| `PROFILE=1` | Generate Chrome trace file |
| `VIZ=1` | Interactive UOp graph visualization |
| `NOOPT=1` | Disable kernel optimization |

### Key Results
| Metric | Value |
|--------|-------|
| tinygrad (BEAM cached) | **36.71 tok/s** |
| llama.cpp (CUDA) | 25.59–27.80 tok/s |
| Theoretical max (BW-limited) | ~93 tok/s |
| Matvec heuristic speedup | 7.6× |
| Model | Qwen3 1B Q6_K |

## Philosophy

TinyGrad prioritizes:
1. **Simplicity** — ~10K lines total, every line earns its keep
2. **Hackability** — The whole stack is visible and modifiable
3. **Performance** — Generated kernels beat hand-tuned C++
4. **Education** — Code that teaches as you read it

## Contributing

Found an error or want to improve a pill? PRs welcome! Keep the tl;dr style — dense with information but still readable.

## License

Same as TinyGrad (MIT)
