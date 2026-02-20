# Pill 16: Jetson DevKit Practical Guide

Everything you need to know to get the most out of your Jetson AGX Orin 64GB for LLM inference and ML workloads. This pill is the hands-on companion to Pill 15's hardware theory.

---

## Part 1: Power Modes and Clock Management

### nvpmodel: Choosing Your Power Profile

The Orin SoC has configurable power/performance profiles. Each mode caps TDP, clock speeds, and active CPU cores. Pick the right one for your use case:

```
┌──────┬──────────┬────────────────────┬───────────┬───────────────────┐
│ Mode │ Name     │ CPU                │ GPU MHz   │ TDP (approx)      │
├──────┼──────────┼────────────────────┼───────────┼───────────────────┤
│  0   │ MAXN     │ 12 cores @ 2.2GHz │ 1300      │ ~60W              │
│  1   │ 50W      │ 12 cores @ 2.2GHz │ 1300      │  50W (throttled)  │
│  2   │ 30W 12c  │ 12 cores @ 1.5GHz │  930      │ ~30W              │
│  3   │ 30W 8c   │ 8 cores  @ 1.5GHz │  930      │ ~30W              │
│  4   │ 30W 6c   │ 6 cores  @ 1.5GHz │  930      │ ~30W              │
│  5   │ 30W 4c   │ 4 cores  @ 1.5GHz │  930      │ ~30W              │
│  6   │ 15W      │ 4 cores  @ 1.2GHz │  510      │ ~15W              │
│  7   │ 15W 2c   │ 2 cores  @ 1.2GHz │   510      │ ~15W              │
└──────┴──────────┴────────────────────┴───────────┴───────────────────┘

Note: Exact modes vary by JetPack version and board SKU. Use `nvpmodel -p` to list all available modes on your specific board.
```

**Essential commands:**

```bash
# Check current power mode
sudo nvpmodel -q

# Switch to MAXN (maximum performance)
sudo nvpmodel -m 0

# List all available modes and their configs
sudo nvpmodel -p

# After switching mode, lock clocks to maximum within that TDP:
sudo jetson_clocks

# Show current clock frequencies
sudo jetson_clocks --show
```

**When to use each mode:**

```
Benchmarking / maximum tok/s:
  → MODE 0 (MAXN) + sudo jetson_clocks
  → This is what all numbers in these pills use

24/7 robot running Q0.6B for real-time decisions:
  → MODE 2 or 3 (30W)
  → Still ~18 tok/s on LLaMA 1B, plenty for real-time NLP

Battery-powered / thermal-constrained (drone, handheld):
  → MODE 6 or 7 (15W)
  → ~8 tok/s on LLaMA 1B — still useful for small models

Development / compiling:
  → MODE 0 or 1
  → NixOS rebuilds benefit from all 12 CPU cores
```

### jetson_clocks: Locking Clock Frequencies

By default, Orin uses dynamic frequency scaling (DVFS) — clocks ramp up/down based on load. This is great for power efficiency but introduces variance in benchmarks:

```bash
# Lock clocks to maximum (within current nvpmodel TDP)
sudo jetson_clocks

# Check what clocks are set to
sudo jetson_clocks --show

# Example output:
#   SOC family: tegra234
#   Online CPUs: 0-11
#   cpu0: Online=1 Governor=performance MinFreq=729600 MaxFreq=2201600 CurrentFreq=2201600
#   ...
#   GPU MinFreq=306000000 MaxFreq=1300500000 CurrentFreq=1300500000
#   EMC MaxFreq=3199000000 CurrentFreq=3199000000 FreqOverride=1

# Restore dynamic scaling
sudo jetson_clocks --restore
```

**Important:** `jetson_clocks` pins clocks within the *current* nvpmodel TDP. If you're in MODE 6 (15W), it locks GPU at 510 MHz, not 1300 MHz. Always set nvpmodel FIRST, then jetson_clocks.

---

## Part 2: Monitoring with tegrastats

`tegrastats` is the Orin's equivalent of `nvidia-smi` + `htop` combined. It shows real-time CPU, GPU, memory, power, and thermal data:

```bash
# Start monitoring (prints one line per second)
sudo tegrastats

# Sample output (annotated):
# RAM 8234/62843MB (lfb 11939x4MB)   ← 8.2 GB used of 62.8 GB, largest free block 47.7 GB
# SWAP 0/31422MB (cached 0MB)        ← swap usage (should be 0 for perf)
# CPU [20%@2201,15%@2201,...]        ← per-core utilization @ freq (MHz)
# GR3D_FREQ 99%@1300                 ← GPU utilization @ clock (99% = fully busy!)
# VIC_FREQ 0%@115                    ← Video Image Compositor
# APE 174                            ← Audio Processing Engine freq
# CV0@52.5C CPU@48.8C ...            ← Thermal zones (Celsius)
# SOC 52.312C GPU 50.75C ...         ← More thermal sensors
# VDD_GPU_SOC 15432mW                ← GPU+SoC rail power (milliwatts)
# VDD_CPU_CV 3118mW                  ← CPU+CV rail power
# VIN_SYS_5V0 23937mW               ← Total system input power
# VDDRQ 4312mW                       ← DRAM power

# Set update interval (milliseconds)
sudo tegrastats --interval 200    # 5 updates per second

# Log to file for later analysis
sudo tegrastats --interval 1000 --logfile /tmp/perf.log
```

### Reading tegrastats During LLM Inference

Here's what the numbers look like during tinygrad decode:

```
Typical tinygrad NV=1 decode (LLaMA 1B fp16, MODE 0):

  GR3D_FREQ 98%@1300     ← GPU nearly maxed out (memory-bound, not idle)
  RAM 12400/62843MB       ← ~12 GB used (model + activations + system)
  CPU [8%@2201, ...]      ← CPU mostly idle (NV backend uses direct MMIO)
  VDD_GPU_SOC 22500mW     ← GPU+SoC pulling ~22.5W
  VDD_CPU_CV   1800mW     ← CPU barely working
  VIN_SYS_5V0 28000mW     ← System total ~28W (under 60W TDP)
  GPU@55.2C               ← GPU temperature (comfortable, fan running)

Typical llama.cpp CUDA decode (same model):

  GR3D_FREQ 75%@1300     ← GPU less utilized (dispatch gaps!)
  CPU [35%@2201, ...]     ← CPU doing more work (CUDA runtime overhead)
  VDD_GPU_SOC 19000mW     ← Less GPU power (less utilized)
  VDD_CPU_CV   5500mW     ← Significantly more CPU power
```

**Key insight:** tinygrad's NV backend shows *higher GPU utilization* and *lower CPU utilization* than CUDA-based frameworks. The GPU is doing more useful work per clock cycle.

### Automated Power Measurement Script

```bash
#!/bin/bash
# measure-power.sh — Average power during a benchmark run
# Usage: ./measure-power.sh <command_to_benchmark>

LOGFILE=/tmp/power-$$.log

# Start tegrastats in background
sudo tegrastats --interval 100 --logfile "$LOGFILE" &
TEGRA_PID=$!

# Run the actual benchmark
"$@"

# Stop tegrastats
sudo kill $TEGRA_PID 2>/dev/null
sleep 0.5

# Parse and compute averages
echo ""
echo "=== Power Summary ==="
awk -F'[ /]' '
  /VIN_SYS_5V0/ {
    match($0, /VIN_SYS_5V0 ([0-9]+)mW/, a)
    if (a[1]) { total += a[1]; n++ }
  }
  END {
    if (n > 0) printf "Average system power: %.1f W  (%d samples)\n", total/n/1000, n
  }
' "$LOGFILE"

awk -F'[ /]' '
  /GR3D_FREQ/ {
    match($0, /GR3D_FREQ ([0-9]+)%/, a)
    if (a[1]) { total += a[1]; n++ }
  }
  END {
    if (n > 0) printf "Average GPU utilization: %.1f%%  (%d samples)\n", total/n, n
  }
' "$LOGFILE"

rm -f "$LOGFILE"
```

---

## Part 3: Thermal Management

### Temperature Zones

The Orin has multiple thermal sensors. During LLM inference, the ones that matter:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Orin AGX Thermal Map                          │
│                                                                  │
│  ┌─────────┐                      ┌─────────────┐              │
│  │  CPU    │  CPU@48°C            │   GPU        │ GPU@55°C    │
│  │ cluster │  (usually cool       │  (the hot    │             │
│  │         │   during inference)   │   one!)      │             │
│  └─────────┘                      └─────────────┘              │
│                                                                  │
│  ┌─────────┐    ┌─────────────┐                                │
│  │  SOC   │    │   DRAM       │   DRAM temp not reported       │
│  │ 52°C   │    │  (under SoC) │   separately; SoC temp is     │
│  └─────────┘    └─────────────┘   the proxy                    │
│                                                                  │
│  Thermal trip points (auto-throttle):                           │
│    GPU:  throttle at ~90°C,  shutdown at 105°C                  │
│    CPU:  throttle at ~90°C,  shutdown at 105°C                  │
│    SoC:  throttle at ~90°C                                      │
│                                                                  │
│  Comfortable operating range for sustained inference:           │
│    GPU: 45-70°C  ✓  (fan curve keeps it here)                   │
│    GPU: 70-80°C  ⚠  (thermal headroom shrinking)                │
│    GPU: 80-90°C  ⚡  (approaching throttle)                     │
└──────────────────────────────────────────────────────────────────┘
```

### Monitoring Temperature

```bash
# Via tegrastats (simplest)
sudo tegrastats | grep -oP '(CPU|GPU|SOC)@[\d.]+C'

# Direct sysfs reads (if tegrastats isn't available)
cat /sys/devices/virtual/thermal/thermal_zone*/type
cat /sys/devices/virtual/thermal/thermal_zone*/temp
# temp values are in millidegrees: 55250 = 55.25°C

# One-liner: show all zones with names and temps
paste <(cat /sys/devices/virtual/thermal/thermal_zone*/type) \
      <(cat /sys/devices/virtual/thermal/thermal_zone*/temp) | \
  awk '{printf "%-20s %.1f°C\n", $1, $2/1000}'
```

### Fan Control

The Jetson AGX Orin devkit has an active fan with automatic thermal management:

```bash
# Check current fan speed (0-255)
cat /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1

# Fan mode: 0 = manual, 1 = auto (thermal-controlled)
cat /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1_enable

# For benchmarking, you might want max fan speed to prevent throttling:
echo 255 | sudo tee /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1

# Return to automatic:
echo 1 | sudo tee /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1_enable
```

**Thermal throttling diagnosis:** If your tok/s drops during a long benchmark, check if `GR3D_FREQ` in tegrastats shows a lower clock than your nvpmodel maximum. If GPU@>85°C and clock dropped from 1300 to 1100 or lower, you're throttling.

---

## Part 4: Memory Management on Unified Memory

### Understanding Unified Memory

The Orin's 64 GB LPDDR5 is shared between CPU and GPU — there is no separate "VRAM". This has practical implications:

```
Total memory:      64 GB LPDDR5
System reserved:   ~1.2 GB (kernel, firmware, display)
Available:         ~62.8 GB

Memory users (during LLM inference):
  NixOS system:          ~1.5 GB
  tinygrad runtime:      ~0.2 GB
  Model weights (fp16):  varies by model
    Qwen3 0.6B:         ~1.2 GB
    LLaMA 1B:           ~2.0 GB
    LLaMA 3B:           ~6.0 GB
  KV cache:              model-dependent, grows with context
  CUDA/NV driver:        ~0.3 GB
  ─────────────────────────────────
  Free for more models:  ~52+ GB (!)

  This means you can load MUCH bigger models than on a 24 GB RTX 4090.
  LLaMA 70B Q4 (~35 GB) fits in memory! (slowly, but it fits)
```

### Monitoring Memory

```bash
# System memory overview
free -h

# Per-process GPU memory (via /proc)
# On Jetson, GPU memory IS system memory, so just use:
ps aux --sort=-%mem | head -20

# NVIDIA-specific memory stats (if using CUDA):
cat /sys/kernel/debug/nvmap/iovmm/clients

# Total GPU memory mapped by a process:
cat /proc/$(pgrep -f tinygrad)/maps | grep -c nvidia
```

### The Unified Memory Advantage for tinygrad

```
Desktop GPU workflow:
  1. CPU loads model from disk to CPU RAM     (disk → DDR5)
  2. CPU copies weights to GPU VRAM           (DDR5 → PCIe → GDDR6X)
  3. GPU runs inference from VRAM             (GDDR6X → SMs)
  Step 2 is wasted time and wasted power.

Jetson Orin workflow:
  1. CPU loads model from disk to LPDDR5      (disk → LPDDR5)
  2. GPU runs inference directly from LPDDR5  (LPDDR5 → SMs)
  No step 2! Zero-copy.

  For mmap'd models (how tinygrad loads GGUF):
  1. mmap() maps the file into the address space
  2. First GPU access triggers page fault → pages loaded from disk
  3. Subsequent accesses hit page cache — already in LPDDR5
  → Model "loading" time is essentially zero (deferred to first use)
```

---

## Part 5: DLA (Deep Learning Accelerator)

The Orin has two DLA cores that most people ignore. They're specialized INT8/FP16 inference engines separate from the GPU:

```
DLA quick facts:
  - 2× DLA v2.0 cores on Orin AGX 64GB
  - Combined INT8 throughput: up to ~170 TOPs
  - Support: convolution, deconvolution, pooling, LRN, concat, element-wise
  - Do NOT support: general matmul, attention, most LLM ops
  - Frameworks: TensorRT (primary), via ONNX-TRT conversion
  - NOT useful for LLM inference (no transformer support)

When DLA makes sense:
  - CNN-based computer vision running alongside LLM inference
  - Object detection (YOLO, SSD) at INT8 on DLA
    while LLM runs on GPU simultaneously
  - Always-on camera processing (person detection, lane tracking)
    at ~5W for the DLA portion alone
  - Sensor fusion pipelines in robotics

When DLA doesn't help:
  - LLM inference (no attention/matmul for large arbitrary shapes)
  - tinygrad workloads (no DLA backend in tinygrad)
  - Training (DLA is inference-only)
```

```bash
# Check DLA availability
ls /dev/nvdla*
# Should show /dev/nvdla0 and /dev/nvdla1

# DLA is accessed through TensorRT — requires model compilation:
# trtexec --onnx=model.onnx --useDLACore=0 --fp16
```

---

## Part 6: Storage and Model Loading

### NVMe vs eMMC vs SD Card

Model loading speed depends on your storage. The devkit has multiple options:

```
┌──────────────┬────────────┬────────────────────────────────────┐
│ Storage      │ Read Speed │ Time to load LLaMA 3B fp16 (6 GB) │
├──────────────┼────────────┼────────────────────────────────────┤
│ NVMe SSD     │ ~3 GB/s    │ ~2 seconds                         │
│ eMMC (onbd)  │ ~250 MB/s  │ ~24 seconds                        │
│ SD card      │ ~90 MB/s   │ ~67 seconds                        │
│ USB 3.2 SSD  │ ~800 MB/s  │ ~7.5 seconds                       │
└──────────────┴────────────┴────────────────────────────────────┘

Recommendation: Always use NVMe for model storage.
The devkit has an M.2 NVMe slot (Key M, 2280).
```

### Optimizing Model Load Time

```bash
# Check your storage device and filesystem
df -Th /path/to/models/

# Pre-warm the page cache (useful before benchmarks):
# This reads the entire file into memory without processing it
cat /path/to/model.gguf > /dev/null

# Verify it's cached:
vmtouch /path/to/model.gguf
# Output shows percentage in page cache (should be 100% after cat)

# For repeated benchmarks, the model stays in page cache
# between runs — only first load is slow
```

---

## Part 7: Practical Tips for Orin LLM Development

### Pre-Benchmark Checklist

```bash
# 1. Set maximum performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. Verify clocks are locked
sudo jetson_clocks --show | grep -E 'GPU|EMC'
# Should show: GPU CurrentFreq=1300500000, EMC CurrentFreq=3199000000

# 3. Check temperature (start cool)
sudo tegrastats --interval 500 &
# GPU should be <50°C before starting

# 4. Kill background processes that eat memory/CPU
sudo systemctl stop docker containerd 2>/dev/null
# (restart later: sudo systemctl start docker)

# 5. Pre-warm the model file in page cache
cat /path/to/model.gguf > /dev/null

# 6. Run the benchmark
NV=1 python3 -m tinygrad.nn.gguf --model /path/to/model.gguf \
  --count 128 --prompt "Hello"

# 7. Stop tegrastats
kill %1
```

### Common Issues and Fixes

```
Problem: tok/s drops after 30+ seconds of inference
Cause:   Thermal throttling
Fix:     Check GPU temp in tegrastats. If >85°C:
         - Ensure fan is running (check pwm1 value)
         - Improve airflow (don't put the devkit in an enclosure)
         - Consider MODE 1 (50W) — slightly less power, much less heat

Problem: "out of memory" with models that should fit
Cause:   Fragmentation or other processes using LPDDR5
Fix:     echo 3 | sudo tee /proc/sys/vm/drop_caches
         Kill unnecessary processes (docker, X11, etc.)
         Check: free -h (should show ~60 GB available)

Problem: Low GPU utilization (GR3D_FREQ <80%) during decode
Cause:   Dispatch overhead (CUDA runtime) or Python overhead
Fix:     Use NV=1 (tinygrad NV backend) instead of CUDA runtime
         Verify with: NV=1 DEBUG=1 python3 ...

Problem: "Permission denied" accessing /dev/nvhost-*
Cause:   Missing udev rules or user not in video group
Fix:     sudo usermod -aG video $USER
         Or on NixOS: users.users.agent.extraGroups = [ "video" ];

Problem: nvpmodel: command not found (NixOS)
Cause:   NixOS doesn't ship NVIDIA's userspace tools by default
Fix:     The jetpack-nixos flake includes these via modules.
         Check: nix shell nixpkgs#pciutils -c lspci | grep NVIDIA
         Verify GPU is detected, then check your NixOS config.

Problem: EMC (memory) clock not at max after jetson_clocks
Cause:   Sometimes needs a second invocation
Fix:     sudo jetson_clocks && sleep 1 && sudo jetson_clocks
         Verify: sudo jetson_clocks --show | grep EMC
```

### NixOS-Specific Configuration

On NixOS (which is what we run on this devkit), some Jetson-specific features need explicit enablement:

```nix
# In your NixOS configuration (e.g., configuration.nix):
{
  # Enable NVIDIA GPU
  hardware.nvidia-jetpack.enable = true;

  # Set power mode at boot (optional)
  # systemd.services.nvpmodel = {
  #   after = [ "multi-user.target" ];
  #   wantedBy = [ "multi-user.target" ];
  #   serviceConfig.ExecStart = "${pkgs.nvpmodel}/bin/nvpmodel -m 0";
  # };

  # Ensure video group access for GPU devices
  users.users.agent.extraGroups = [ "video" "render" ];

  # Fan control service (jetpack-nixos module)
  hardware.nvidia-jetpack.nvfancontrol.enable = true;
}
```

---

## Part 8: Quick Reference Card

### The Numbers That Matter

```
┌───────────────────────────────────────────────────────────────────┐
│               Jetson AGX Orin 64GB — Quick Reference              │
├───────────────────────────────────────────────────────────────────┤
│ GPU:        2048 CUDA cores, 64 Tensor Cores, 16 SMs            │
│ CPU:        12× Arm Cortex-A78AE @ 2.2 GHz                      │
│ Memory:     64 GB LPDDR5, ~102 GB/s read BW (unified)           │
│ DLA:        2× DLA v2.0 (INT8, for CNN inference)                │
│ SM:         8.7 (Ampere-based)                                   │
│ TDP:        15W – 60W (nvpmodel configurable)                    │
│ Process:    Samsung 8nm                                          │
│ JetPack:    6.x (L4T r36.4.x, CUDA 12.6)                       │
├───────────────────────────────────────────────────────────────────┤
│ Peak FP16 Tensor:           5.3 TFLOPS                           │
│ Peak FP32 (non-Tensor):     2.6 TFLOPS                           │
│ Peak INT8 Tensor:          10.6 TOPS (GPU) + 170 TOPS (DLA)     │
│ Memory BW (realistic):     ~102 GB/s read                        │
│ L3 cache:                   4 MB (system-wide)                   │
│ SMEM per SM:                up to 228 KB (configurable)          │
│ Register file per SM:       256 KB (65,536 × 32-bit)            │
├───────────────────────────────────────────────────────────────────┤
│ LLM decode tok/s (tinygrad NV=1, MODE 0):                       │
│   Qwen3 0.6B fp16:   41.0    LLaMA 1B fp16:    29.0            │
│   LLaMA 3B fp16:     12.1    Max model (64GB):  ~LLaMA 30B Q4  │
├───────────────────────────────────────────────────────────────────┤
│ Useful commands:                                                  │
│   sudo nvpmodel -m 0          (max performance)                  │
│   sudo jetson_clocks           (lock clocks)                     │
│   sudo tegrastats              (real-time monitoring)            │
│   sudo jetson_clocks --show    (verify clocks)                   │
│   free -h                      (memory usage)                    │
└───────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [← Pill 15: Why TinyGrad Wins and Loses](15-why-tinygrad-wins-and-loses.md)
