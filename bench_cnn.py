#!/usr/bin/env python3
"""
CNN Benchmark Suite for tinygrad on Jetson Orin AGX.

Tests convolutional neural network workloads to compare NV=1 vs CUDA=1:
  - Individual layer benchmarks (Conv2d, BatchNorm, ReLU, MaxPool)
  - Full model inference (ResNet-like, EfficientNet-like architectures)
  - Different input resolutions (224x224, 640x640 for detection)
  - fp32 vs fp16 precision comparison

This is a pure-tinygrad benchmark: no pretrained weights needed.
We construct models from scratch and benchmark forward pass throughput.

Usage:
    NV=1 python3 bench_cnn.py
    CUDA=1 python3 bench_cnn.py
    NV=1 JITBEAM=2 python3 bench_cnn.py
    NV=1 HALF=1 python3 bench_cnn.py          # fp16 mode
"""
import os, time, sys, gc

if __name__ != "__main__":
    raise RuntimeError("Run this script directly, not via import")

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Conv2d, BatchNorm, Linear

backend = "NV" if os.environ.get("NV") else "CUDA" if os.environ.get("CUDA") else "CPU"
jitbeam = os.environ.get("JITBEAM", "0")
use_half = bool(os.environ.get("HALF"))
dt = dtypes.float16 if use_half else dtypes.float32
dt_name = "fp16" if use_half else "fp32"

print(f"\n{'='*70}")
print(f"CNN Benchmark Suite - {backend}=1  JITBEAM={jitbeam}  {dt_name}")
print(f"Device: {Device.DEFAULT} | Jetson Orin AGX 64GB")
print(f"{'='*70}\n")

# ============= UTILITIES =============

def time_fn(fn, warmup=3, iters=10):
    """Time a function with warmup. Returns (avg_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.time()
        fn()
        times.append((time.time() - t0) * 1000)
    avg = sum(times) / len(times)
    return avg, min(times), max(times)

def fmt_result(name, avg_ms, min_ms, max_ms, extra=""):
    """Format a single benchmark result."""
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    print(f"  {name:<40} {avg_ms:7.2f}ms  ({min_ms:.1f}-{max_ms:.1f}ms)  {fps:6.1f} FPS  {extra}")
    return {"name": name, "avg_ms": round(avg_ms, 2), "fps": round(fps, 1)}

# ============= PART 1: INDIVIDUAL LAYER BENCHMARKS =============

print("Part 1: Individual Layer Performance")
print("─" * 70)
print("  (Batch=1, 224x224 input, common layer configs)\n")

layer_results = []

# Conv2d layers (most compute-intensive in CNNs)
conv_configs = [
    ("Conv2d 3->64 k7s2",    3,  64, 7, 2, 3),   # ResNet stem
    ("Conv2d 64->64 k3s1",  64,  64, 3, 1, 1),    # ResNet block
    ("Conv2d 64->128 k3s2", 64, 128, 3, 2, 1),    # Downsample
    ("Conv2d 128->256 k3s1",128, 256, 3, 1, 1),   # Deep layer
    ("Conv2d 256->512 k3s1",256, 512, 3, 1, 1),   # Deepest
    ("Conv2d 32->32 k3s1 DW", 32, 32, 3, 1, 1),   # Depthwise (MobileNet style)
]

for name, cin, cout, k, s, p in conv_configs:
    try:
        # Determine input spatial based on stride accumulation
        spatial = 224 // (2 if s == 2 else 1)
        if cin >= 128: spatial = 56
        if cin >= 256: spatial = 28

        groups = cin if "DW" in name else 1
        conv = Conv2d(cin, cout, k, stride=s, padding=p, groups=groups)
        x = Tensor.randn(1, cin, spatial, spatial, dtype=dt)

        def run():
            y = conv(x)
            y.realize()

        avg, mn, mx = time_fn(run, warmup=3, iters=10)
        r = fmt_result(name, avg, mn, mx)
        layer_results.append(r)
        del conv, x
    except Exception as e:
        print(f"  {name:<40} ERROR: {e}")

# BatchNorm
for channels in [64, 128, 256]:
    spatial = 56 if channels <= 64 else 28
    try:
        # BatchNorm in tinygrad: just scale + bias + running stats
        weight = Tensor.ones(channels, dtype=dt)
        bias = Tensor.zeros(channels, dtype=dt)
        x = Tensor.randn(1, channels, spatial, spatial, dtype=dt)
        def run():
            # Manual batchnorm: (x - mean) / sqrt(var + eps) * weight + bias
            y = (x - x.mean(axis=(2,3), keepdim=True)) * weight.reshape(1,-1,1,1) + bias.reshape(1,-1,1,1)
            y.realize()
        avg, mn, mx = time_fn(run, warmup=3, iters=10)
        r = fmt_result(f"BatchNorm {channels}ch {spatial}x{spatial}", avg, mn, mx)
        layer_results.append(r)
        del weight, bias, x
    except Exception as e:
        print(f"  BatchNorm {channels}ch ERROR: {e}")

# MaxPool
for pool_size in [2, 3]:
    try:
        x = Tensor.randn(1, 64, 112, 112, dtype=dt)
        def run():
            y = x.max_pool2d(kernel_size=pool_size, stride=pool_size)
            y.realize()
        avg, mn, mx = time_fn(run, warmup=3, iters=10)
        r = fmt_result(f"MaxPool{pool_size}x{pool_size} 64ch 112x112", avg, mn, mx)
        layer_results.append(r)
        del x
    except Exception as e:
        print(f"  MaxPool{pool_size} ERROR: {e}")

# ReLU
for channels in [64, 256]:
    spatial = 56
    try:
        x = Tensor.randn(1, channels, spatial, spatial, dtype=dt)
        def run():
            y = x.relu()
            y.realize()
        avg, mn, mx = time_fn(run, warmup=3, iters=10)
        r = fmt_result(f"ReLU {channels}ch {spatial}x{spatial}", avg, mn, mx)
        layer_results.append(r)
        del x
    except Exception as e:
        print(f"  ReLU {channels}ch ERROR: {e}")

# Linear (FC layer at end of CNN)
for in_feat, out_feat in [(512, 1000), (2048, 1000)]:
    try:
        fc = Linear(in_feat, out_feat)
        x = Tensor.randn(1, in_feat, dtype=dt)
        def run():
            y = fc(x)
            y.realize()
        avg, mn, mx = time_fn(run, warmup=3, iters=10)
        r = fmt_result(f"Linear {in_feat}->{out_feat}", avg, mn, mx)
        layer_results.append(r)
        del fc, x
    except Exception as e:
        print(f"  Linear {in_feat}->{out_feat} ERROR: {e}")

gc.collect()

# ============= PART 2: FULL MODEL INFERENCE =============

print(f"\n\nPart 2: Full Model Inference (Batch=1)")
print("─" * 70)

model_results = []

# --- ResNet-18-like ---
class ResBlock:
    def __init__(self, cin, cout, stride=1):
        self.conv1 = Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.conv2 = Conv2d(cout, cout, 3, stride=1, padding=1)
        self.downsample = Conv2d(cin, cout, 1, stride=stride) if cin != cout or stride != 1 else None
    def __call__(self, x):
        residual = x
        out = self.conv1(x).relu()
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return (out + residual).relu()

class MiniResNet:
    """ResNet-18-like: 11.7M params"""
    def __init__(self, num_classes=1000):
        self.conv1 = Conv2d(3, 64, 7, stride=2, padding=3)
        # 4 stages: 64, 128, 256, 512 channels
        self.layer1 = [ResBlock(64, 64), ResBlock(64, 64)]
        self.layer2 = [ResBlock(64, 128, stride=2), ResBlock(128, 128)]
        self.layer3 = [ResBlock(128, 256, stride=2), ResBlock(256, 256)]
        self.layer4 = [ResBlock(256, 512, stride=2), ResBlock(512, 512)]
        self.fc = Linear(512, num_classes)

    def __call__(self, x):
        x = self.conv1(x).relu()
        x = x.max_pool2d(kernel_size=3, stride=2, padding=1)
        for b in self.layer1: x = b(x)
        for b in self.layer2: x = b(x)
        for b in self.layer3: x = b(x)
        for b in self.layer4: x = b(x)
        x = x.mean(axis=(2, 3))  # Global average pool
        return self.fc(x)

try:
    print(f"\n  ResNet-18-like (11.7M params, 224x224 input):")
    resnet = MiniResNet(num_classes=1000)
    x = Tensor.randn(1, 3, 224, 224, dtype=dt)
    def run():
        y = resnet(x)
        y.realize()
    avg, mn, mx = time_fn(run, warmup=3, iters=10)
    r = fmt_result("ResNet-18-like 224x224", avg, mn, mx)
    model_results.append(r)
    del resnet, x
    gc.collect()
except Exception as e:
    print(f"  ResNet-18-like ERROR: {e}")

# --- MobileNet-v1-like (depthwise separable) ---
class DWSepConv:
    """Depthwise separable convolution (MobileNet building block)"""
    def __init__(self, cin, cout, stride=1):
        self.dw = Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin)
        self.pw = Conv2d(cin, cout, 1)
    def __call__(self, x):
        return self.pw(self.dw(x).relu()).relu()

class MiniMobileNet:
    """MobileNet-v1-like: ~3.4M params, optimized for mobile/edge"""
    def __init__(self, num_classes=1000):
        self.conv1 = Conv2d(3, 32, 3, stride=2, padding=1)
        self.layers = [
            DWSepConv(32, 64),
            DWSepConv(64, 128, stride=2),
            DWSepConv(128, 128),
            DWSepConv(128, 256, stride=2),
            DWSepConv(256, 256),
            DWSepConv(256, 512, stride=2),
            DWSepConv(512, 512),
            DWSepConv(512, 512),
            DWSepConv(512, 512),
            DWSepConv(512, 512),
            DWSepConv(512, 512),
            DWSepConv(512, 1024, stride=2),
            DWSepConv(1024, 1024),
        ]
        self.fc = Linear(1024, num_classes)

    def __call__(self, x):
        x = self.conv1(x).relu()
        for layer in self.layers:
            x = layer(x)
        x = x.mean(axis=(2, 3))
        return self.fc(x)

try:
    print(f"\n  MobileNet-v1-like (3.4M params, 224x224 input):")
    mobilenet = MiniMobileNet(num_classes=1000)
    x = Tensor.randn(1, 3, 224, 224, dtype=dt)
    def run():
        y = mobilenet(x)
        y.realize()
    avg, mn, mx = time_fn(run, warmup=3, iters=10)
    r = fmt_result("MobileNet-v1-like 224x224", avg, mn, mx)
    model_results.append(r)
    del mobilenet, x
    gc.collect()
except Exception as e:
    print(f"  MobileNet-v1-like ERROR: {e}")

# --- YOLO-like detection backbone (640x640 input) ---
class YOLOBackbone:
    """Simplified YOLO-like backbone for 640x640 detection input"""
    def __init__(self):
        self.stem = Conv2d(3, 32, 3, stride=2, padding=1)
        self.stage1 = [Conv2d(32, 64, 3, stride=2, padding=1)]
        self.stage2 = [Conv2d(64, 128, 3, stride=2, padding=1), Conv2d(128, 128, 3, padding=1)]
        self.stage3 = [Conv2d(128, 256, 3, stride=2, padding=1), Conv2d(256, 256, 3, padding=1)]
        self.stage4 = [Conv2d(256, 512, 3, stride=2, padding=1), Conv2d(512, 512, 3, padding=1)]
    def __call__(self, x):
        x = self.stem(x).relu()
        for c in self.stage1: x = c(x).relu()
        for c in self.stage2: x = c(x).relu()
        for c in self.stage3: x = c(x).relu()
        for c in self.stage4: x = c(x).relu()
        return x

try:
    print(f"\n  YOLO-like Backbone (640x640 detection input):")
    yolo = YOLOBackbone()
    x = Tensor.randn(1, 3, 640, 640, dtype=dt)
    def run():
        y = yolo(x)
        y.realize()
    avg, mn, mx = time_fn(run, warmup=3, iters=10)
    r = fmt_result("YOLO-backbone 640x640", avg, mn, mx)
    model_results.append(r)
    del yolo, x
    gc.collect()
except Exception as e:
    print(f"  YOLO-backbone ERROR: {e}")

# --- Tiny inference model (edge deployment) ---
class TinyConvNet:
    """Ultra-small CNN for real-time edge inference (~50K params)"""
    def __init__(self, num_classes=10):
        self.conv1 = Conv2d(3, 16, 3, padding=1)
        self.conv2 = Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = Conv2d(64, 64, 3, stride=2, padding=1)
        self.fc = Linear(64, num_classes)
    def __call__(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = x.mean(axis=(2, 3))  # Global average pool
        return self.fc(x)

try:
    print(f"\n  TinyConvNet (~50K params, 128x128 edge inference):")
    tiny = TinyConvNet(num_classes=10)
    x = Tensor.randn(1, 3, 128, 128, dtype=dt)
    def run():
        y = tiny(x)
        y.realize()
    avg, mn, mx = time_fn(run, warmup=3, iters=8)
    r = fmt_result("TinyConvNet 128x128", avg, mn, mx)
    model_results.append(r)
    del tiny, x
    gc.collect()
except Exception as e:
    print(f"  TinyConvNet ERROR: {e}")

# ============= PART 3: BATCH SIZE SCALING =============

print(f"\n\nPart 3: Batch Size Scaling (ResNet-18-like)")
print("─" * 70)

batch_results = []
for batch in [1, 2, 4, 8]:
    try:
        resnet = MiniResNet(num_classes=1000)
        x = Tensor.randn(batch, 3, 224, 224, dtype=dt)
        def run():
            y = resnet(x)
            y.realize()
        avg, mn, mx = time_fn(run, warmup=2, iters=5)
        imgs_per_s = batch * 1000.0 / avg
        r = fmt_result(f"batch={batch}", avg, mn, mx, f"({imgs_per_s:.1f} imgs/s)")
        r["batch"] = batch
        r["imgs_per_s"] = round(imgs_per_s, 1)
        batch_results.append(r)
        del resnet, x
        gc.collect()
    except Exception as e:
        print(f"  batch={batch} ERROR: {e}")
        break  # Likely OOM, stop trying larger batches

# ============= SUMMARY =============

print(f"\n{'='*70}")
print(f"SUMMARY  ({backend}=1  JITBEAM={jitbeam}  {dt_name})")
print(f"{'='*70}")

if model_results:
    print(f"\nFull Model Inference (Batch=1):")
    print(f"{'Model':<40} {'Latency':>10} {'FPS':>10}")
    print("─" * 65)
    for r in model_results:
        print(f"{r['name']:<40} {r['avg_ms']:8.2f}ms {r['fps']:8.1f}")

if batch_results:
    print(f"\nBatch Scaling (ResNet-18-like, 224x224):")
    print(f"{'Batch':>6} {'Latency':>10} {'Throughput':>15} {'Scaling':>10}")
    print("─" * 45)
    base_ips = batch_results[0]["imgs_per_s"] if batch_results else 1
    for r in batch_results:
        scale = r["imgs_per_s"] / base_ips if base_ips > 0 else 0
        print(f"{r['batch']:>6} {r['avg_ms']:8.2f}ms {r['imgs_per_s']:10.1f} img/s {scale:8.2f}x")

print(f"\nKey Takeaways:")
print(f"  - Conv2d dominates CNN compute ({backend} kernel quality matters)")
print(f"  - Depthwise conv (MobileNet) benefits from low dispatch overhead")
print(f"  - 640x640 YOLO input has 8x more pixels = longer latency")
print(f"  - Batch scaling shows GPU utilization improvement")
if use_half:
    print(f"  - fp16 mode: 2x less memory, may increase throughput")
print(f"\n{'='*70}\n")
