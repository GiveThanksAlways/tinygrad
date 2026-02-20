from __future__ import annotations
import os, ctypes, contextlib, re, functools, mmap, struct, array, sys, weakref, fcntl
assert sys.platform != 'win32'
from typing import cast, ClassVar
from dataclasses import dataclass
from tinygrad.runtime.support.hcq import HCQCompiled, HCQAllocator, HCQBuffer, HWQueue, CLikeArgsState, HCQProgram, HCQSignal, BumpAllocator
from tinygrad.runtime.support.hcq import MMIOInterface, FileIOInterface, MOCKGPU, hcq_filter_visible_devices, hcq_profile
from tinygrad.uop.ops import sint
from tinygrad.device import Compiled, BufferSpec, CompilerSet
from tinygrad.helpers import getenv, mv_address, round_up, data64, data64_le, prod, OSX, to_mv, hi32, lo32, NV_CC, NV_PTX, NV_NAK, NV_NVCC, PROFILE
from tinygrad.helpers import ContextVar, VIZ, ProfileEvent
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.runtime.autogen import nv_570, nv_580, pci, mesa
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager
from tinygrad.runtime.support.system import System, PCIIfaceBase, MAP_FIXED
from tinygrad.renderer.nir import NAKRenderer
if getenv("IOCTL"): import extra.nv_gpu_driver.nv_ioctl # noqa: F401 # pylint: disable=unused-import

nv_gpu = nv_570 # default to 570

PMA = ContextVar("PMA", abs(VIZ.value)>=2)

@dataclass(frozen=True)
class ProfilePMAEvent(ProfileEvent): device:str; kern:str; blob:bytes; exec_tag:int # noqa: E702

class NVSignal(HCQSignal):
  def _sleep(self, time_spent_since_last_sleep_ms:int):
    # Reasonable to sleep for long workloads (which take more than 200ms) and only timeline signals.
    if time_spent_since_last_sleep_ms > 200 and self.is_timeline and self.owner is not None: self.owner.iface.sleep(200)

def get_error_str(status): return f"{status}: {nv_gpu.nv_status_codes.get(status, 'Unknown error')}"

NV_PFAULT_FAULT_TYPE = {dt:name for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_FAULT_TYPE_")}
NV_PFAULT_ACCESS_TYPE = {dt:name.split("_")[-1] for name,dt in nv_gpu.__dict__.items() if name.startswith("NV_PFAULT_ACCESS_TYPE_")}

def nv_flags(reg, **kwargs): return functools.reduce(int.__or__, ((getattr(nv_gpu, f"{reg}_{k}_{v}".upper()) if isinstance(v, str) else v) <<
  getattr(nv_gpu, f"{reg}_{k}".upper())[1] for k, v in kwargs.items()), 0)

def nv_iowr(fd:FileIOInterface, nr, args, cmd=None):
  ret = fd.ioctl(cmd or ((3 << 30) | (ctypes.sizeof(args) & 0x1FFF) << 16 | (ord('F') & 0xFF) << 8 | (nr & 0xFF)), args)
  if ret != 0: raise RuntimeError(f"ioctl returned {ret}")

class QMD:
  fields: dict[str, dict[str, tuple[int, int]]] = {}

  def __init__(self, dev:NVDevice, addr:int|None=None, **kwargs):
    self.ver, self.sz = (5, 0x60) if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else (3, 0x40)

    # Init fields from module
    if (pref:="NVCEC0_QMDV05_00" if self.ver == 5 else "NVC6C0_QMDV03_00") not in QMD.fields:
      QMD.fields[pref] = {**{name[len(pref)+1:]: dt for name,dt in nv_gpu.__dict__.items() if name.startswith(pref) and isinstance(dt, tuple)},
        **{name[len(pref)+1:]+f"_{i}": dt(i) for name,dt in nv_gpu.__dict__.items() for i in range(8) if name.startswith(pref) and callable(dt)}}

    self.mv, self.pref = (memoryview(bytearray(self.sz * 4)) if addr is None else to_mv(addr, self.sz * 4)), pref
    if kwargs: self.write(**kwargs)

  def _rw_bits(self, hi:int, lo:int, value:int|None=None):
    mask = ((1 << (width:=hi - lo + 1)) - 1) << (lo % 8)
    num = int.from_bytes(self.mv[lo//8:hi//8+1], "little")

    if value is None: return (num & mask) >> (lo % 8)

    if value >= (1 << width): raise ValueError(f"{value:#x} does not fit.")
    self.mv[lo//8:hi//8+1] = int((num & ~mask) | ((value << (lo % 8)) & mask)).to_bytes((hi//8 - lo//8 + 1), "little")

  def write(self, **kwargs):
    for k,val in kwargs.items(): self._rw_bits(*QMD.fields[self.pref][k.upper()], value=val) # type: ignore [misc]

  def read(self, k, val=0): return self._rw_bits(*QMD.fields[self.pref][k.upper()])

  def field_offset(self, k): return QMD.fields[self.pref][k.upper()][1] // 8

  def set_constant_buf_addr(self, i, addr):
    if self.ver < 4: self.write(**{f'constant_buffer_addr_upper_{i}':hi32(addr), f'constant_buffer_addr_lower_{i}':lo32(addr)})
    else: self.write(**{f'constant_buffer_addr_upper_shifted6_{i}':hi32(addr >> 6), f'constant_buffer_addr_lower_shifted6_{i}':lo32(addr >> 6)})

class NVCommandQueue(HWQueue[HCQSignal, 'NVDevice', 'NVProgram', 'NVArgsState']):
  def __init__(self):
    self.active_qmd = None
    super().__init__()

  def __del__(self):
    if self.binded_device is not None: self.binded_device.allocator.free(self.hw_page, self.hw_page.size, BufferSpec(cpu_access=True, nolru=True))

  def nvm(self, subchannel, mthd, *args, typ=2): self.q((typ << 28) | (len(args) << 16) | (subchannel << 13) | (mthd >> 2), *args)

  def setup(self, compute_class=None, copy_class=None, local_mem_window=None, shared_mem_window=None, local_mem=None, local_mem_tpc_bytes=None):
    if compute_class: self.nvm(1, nv_gpu.NVC6C0_SET_OBJECT, compute_class)
    if copy_class: self.nvm(4, nv_gpu.NVC6C0_SET_OBJECT, copy_class)
    if local_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A, *data64(local_mem_window))
    if shared_mem_window: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_SHARED_MEMORY_WINDOW_A, *data64(shared_mem_window))
    if local_mem: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_A, *data64(local_mem))
    if local_mem_tpc_bytes: self.nvm(1, nv_gpu.NVC6C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A, *data64(local_mem_tpc_bytes), 0xff)
    return self

  def wait(self, signal:HCQSignal, value:sint=0):
    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="acq_circ_geq", payload_size="64bit"))
    self.active_qmd = None
    return self

  def timestamp(self, signal:HCQSignal): return self.signal(signal, 0)

  def bind(self, dev:NVDevice):
    self.binded_device = dev
    self.hw_page = dev.allocator.alloc(len(self._q) * 4, BufferSpec(cpu_access=True, nolru=True))
    hw_view = self.hw_page.cpu_view().view(fmt='I')
    for i, value in enumerate(self._q): hw_view[i] = value

    # From now on, the queue is on the device for faster submission.
    self._q = hw_view

  def _submit_to_gpfifo(self, dev:NVDevice, gpfifo:GPFifo):
    if dev == self.binded_device: cmdq_addr = self.hw_page.va_addr
    else:
      cmdq_addr = dev.cmdq_allocator.alloc(len(self._q) * 4, 16)
      cmdq_wptr = (cmdq_addr - dev.cmdq_page.va_addr) // 4
      dev.cmdq[cmdq_wptr : cmdq_wptr + len(self._q)] = array.array('I', self._q)

    gpfifo.ring[gpfifo.put_value % gpfifo.entries_count] = (cmdq_addr//4 << 2) | (len(self._q) << 42) | (1 << 41)
    gpfifo.gpput[0] = (gpfifo.put_value + 1) % gpfifo.entries_count

    System.memory_barrier()
    dev.gpu_mmio[0x90 // 4] = gpfifo.token
    gpfifo.put_value += 1

class NVComputeQueue(NVCommandQueue):
  _tegra_signal: ClassVar[bool] = False  # Set True on Tegra to force pushbuffer signal (avoids QMD reuse race)

  def memory_barrier(self):
    self.nvm(1, nv_gpu.NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI,
             nv_flags("NVC6C0_INVALIDATE_SHADER_CACHES_NO_WFI", instruction="true", global_data="true", constant="true"))
    self.active_qmd:QMD|None = None
    return self

  def exec(self, prg:NVProgram, args_state:NVArgsState, global_size:tuple[sint, ...], local_size:tuple[sint, ...]):
    self.bind_args_state(args_state)

    qmd_buf = args_state.buf.offset(round_up(prg.constbufs[0][1], 1 << 8))
    qmd_buf.cpu_view().view(size=prg.qmd.mv.nbytes, fmt='B')[:] = prg.qmd.mv
    assert qmd_buf.va_addr < (1 << 40), f"large qmd addr {qmd_buf.va_addr:x}"

    qmd = QMD(dev=prg.dev, addr=qmd_buf.cpu_view().addr) # Save qmd for later update

    self.bind_sints_to_mem(*global_size, mem=qmd_buf.cpu_view(), fmt='I', offset=qmd.field_offset('cta_raster_width' if qmd.ver<4 else 'grid_width'))
    self.bind_sints_to_mem(*(local_size[:2]), mem=qmd_buf.cpu_view(), fmt='H', offset=qmd.field_offset('cta_thread_dimension0'))
    self.bind_sints_to_mem(local_size[2], mem=qmd_buf.cpu_view(), fmt='B', offset=qmd.field_offset('cta_thread_dimension2'))
    qmd.set_constant_buf_addr(0, args_state.buf.va_addr)

    if self.active_qmd is None:
      if prg.dev.pma_enabled: self.nvm(1, nv_gpu.NVC6C0_PM_TRIGGER, 0)
      self.nvm(1, nv_gpu.NVC6C0_SEND_PCAS_A, qmd_buf.va_addr >> 8)
      self.nvm(1, nv_gpu.NVC6C0_SEND_SIGNALING_PCAS2_B, 9)
    else:
      self.active_qmd.write(dependent_qmd0_pointer=qmd_buf.va_addr >> 8, dependent_qmd0_action=1, dependent_qmd0_prefetch=1, dependent_qmd0_enable=1)

    self.active_qmd, self.active_qmd_buf = qmd, qmd_buf
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    # On Tegra, force pushbuffer-based signal release instead of QMD-based.
    # QMD-based release stores the signal value in shared QMD memory, which gets overwritten by subsequent submits.
    # On Tegra (fast MMIO doorbell), the CPU can submit the next iteration before the GPU reads the current QMD,
    # causing the GPU to read the wrong release value. Pushbuffer-based release avoids this because each submit
    # copies the pushbuffer to a unique offset in cmdq_page (bump-allocated), so signal values are immutable.
    if self._tegra_signal: self.active_qmd = None

    if self.active_qmd is not None:
      for i in range(2):
        if self.active_qmd.read(f'release{i}_enable') == 0:
          self.active_qmd.write(**{f'release{i}_enable': 1})

          addr_off = self.active_qmd.field_offset(f'release{i}_address_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_addr_lower')
          self.bind_sints_to_mem(signal.value_addr & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=addr_off)
          self.bind_sints_to_mem(signal.value_addr >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', mask=0xf, offset=addr_off+4)

          val_off = self.active_qmd.field_offset(f'release{i}_payload_lower' if self.active_qmd.ver<4 else f'release_semaphore{i}_payload_lower')
          self.bind_sints_to_mem(value & 0xffffffff, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off)
          self.bind_sints_to_mem(value >> 32, mem=self.active_qmd_buf.cpu_view(), fmt='I', offset=val_off+4)
          return self

    self.nvm(0, nv_gpu.NVC56F_SEM_ADDR_LO, *data64_le(signal.value_addr), *data64_le(value),
             nv_flags("NVC56F_SEM_EXECUTE", operation="release", release_wfi="en", payload_size="64bit", release_timestamp="en"))
    self.nvm(0, nv_gpu.NVC56F_NON_STALL_INTERRUPT, 0x0)
    self.active_qmd = None
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.compute_gpfifo)

class NVCopyQueue(NVCommandQueue):
  def __init__(self, queue_idx=0):
    self.queue_idx = queue_idx
    super().__init__()

  def copy(self, dest:sint, src:sint, copy_size:int):
    for off in range(0, copy_size, step:=(1 << 31)):
      self.nvm(4, nv_gpu.NVC6B5_OFFSET_IN_UPPER, *data64(src+off), *data64(dest+off))
      self.nvm(4, nv_gpu.NVC6B5_LINE_LENGTH_IN, min(copy_size-off, step))
      self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA,
               nv_flags("NVC6B5_LAUNCH_DMA", data_transfer_type="non_pipelined", src_memory_layout="pitch", dst_memory_layout="pitch"))
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC6B5_SET_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC6B5_LAUNCH_DMA, nv_flags("NVC6B5_LAUNCH_DMA", flush_enable="true", semaphore_type="release_four_word_semaphore"))
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.dma_gpfifo)

class NVVideoQueue(NVCommandQueue):
  def decode_hevc_chunk(self, pic_desc:HCQBuffer, in_buf:HCQBuffer, out_buf:HCQBuffer, out_buf_pos:int, hist_bufs:list[HCQBuffer], hist_pos:list[int],
                        chroma_off:int, coloc_buf:HCQBuffer, filter_buf:HCQBuffer, intra_top_off:int, intra_unk_off:int|None, status_buf:HCQBuffer):
    self.nvm(4, nv_gpu.NVC9B0_SET_APPLICATION_ID, nv_gpu.NVC9B0_SET_APPLICATION_ID_ID_HEVC)
    self.nvm(4, nv_gpu.NVC9B0_SET_CONTROL_PARAMS, nv_flags("NVC9B0_SET_CONTROL_PARAMS", codec_type="hevc", testrun_env="prod_run", gptimer_on=1,
             err_conceal_on=1, mbtimer_on=1, event_trace_logging_on=1))
    self.nvm(4, nv_gpu.NVC9B0_SET_DRV_PIC_SETUP_OFFSET, pic_desc.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_IN_BUF_BASE_OFFSET, in_buf.va_addr >> 8)
    for pos, buf in zip(hist_pos + [out_buf_pos], hist_bufs + [out_buf]):
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_LUMA_OFFSET0 + pos*4, buf.va_addr >> 8)
      self.nvm(4, nv_gpu.NVC9B0_SET_PICTURE_CHROMA_OFFSET0 + pos*4, buf.offset(chroma_off).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_COLOC_DATA_OFFSET, coloc_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_NVDEC_STATUS_OFFSET, status_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_TILE_SIZES_OFFSET, pic_desc.offset(0x200).va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_HEVC_SET_FILTER_BUFFER_OFFSET, filter_buf.va_addr >> 8)
    self.nvm(4, nv_gpu.NVC9B0_SET_INTRA_TOP_BUF_OFFSET, (filter_buf.va_addr + intra_top_off) >> 8)
    if intra_unk_off is not None: self.nvm(4, 0x4dc, (filter_buf.va_addr + intra_unk_off) >> 8)
    self.nvm(4, nv_gpu.NVC9B0_EXECUTE, 0)
    return self

  def signal(self, signal:HCQSignal, value:sint=0):
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_A, *data64(signal.value_addr), value)
    self.nvm(4, nv_gpu.NVC9B0_SEMAPHORE_D, nv_flags("NVC9B0_SEMAPHORE_D", structure_size="four", payload_size="64bit"))
    return self

  def _submit(self, dev:NVDevice): self._submit_to_gpfifo(dev, dev.vid_gpfifo)

class NVArgsState(CLikeArgsState):
  def __init__(self, buf:HCQBuffer, prg:NVProgram, bufs:tuple[HCQBuffer, ...], vals:tuple[int, ...]=()):
    if MOCKGPU: prg.cbuf_0[80:82] = [len(bufs), len(vals)]
    super().__init__(buf, prg, bufs, vals=vals, prefix=prg.cbuf_0 or None)

class NVProgram(HCQProgram):
  def __init__(self, dev:NVDevice, name:str, lib:bytes, **kwargs):
    self.dev, self.name, self.lib = dev, name, lib
    self.constbufs: dict[int, tuple[int, int]] = {0: (0, 0x160)} # dict[constbuf index, tuple[va_addr, size]]

    if (NAK:=isinstance(dev.renderer, NAKRenderer)):
      image, self.cbuf_0 = memoryview(bytearray(lib[ctypes.sizeof(info:=mesa.struct_nak_shader_info.from_buffer_copy(lib)):])), []
      self.regs_usage, self.shmem_usage, self.lcmem_usage = info.num_gprs, round_up(info.cs.smem_size, 128), round_up(info.slm_size, 16)
    elif MOCKGPU: image, sections, relocs = memoryview(bytearray(lib) + b'\x00' * (4 - len(lib)%4)).cast("I"), [], [] # type: ignore
    else: image, sections, relocs = elf_loader(self.lib, force_section_align=128)
    # NOTE: Ensure at least 4KB of space after the program to mitigate prefetch memory faults.
    self.lib_gpu = self.dev.allocator.alloc(round_up((prog_sz:=image.nbytes), 0x1000) + 0x1000, buf_spec:=BufferSpec(nolru=True))
    prog_addr = self.lib_gpu.va_addr
    if not NAK:
      # For MOCKGPU, the lib is PTX code, so some values are emulated.
      self.regs_usage, self.shmem_usage, self.lcmem_usage, cbuf0_size = 0, 0x400, 0x240, 0 if not MOCKGPU else 0x160
      for sh in sections: # pylint: disable=possibly-used-before-assignment
        if sh.name == f".nv.shared.{self.name}": self.shmem_usage = round_up(0x400 + sh.header.sh_size, 128)
        if sh.name == f".text.{self.name}": prog_addr, prog_sz = self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size
        elif m:=re.match(r'\.nv\.constant(\d+)', sh.name):
          self.constbufs[int(m.group(1))] = (self.lib_gpu.va_addr+sh.header.sh_addr, sh.header.sh_size)
        elif sh.name.startswith(".nv.info"):
          for typ, param, data in self._parse_elf_info(sh):
            if sh.name == f".nv.info.{name}" and param == 0xa: cbuf0_size = struct.unpack_from("IH", data)[1] # EIATTR_PARAM_CBANK
            elif sh.name == ".nv.info" and param == 0x12: self.lcmem_usage = struct.unpack_from("II", data)[1] + 0x240 # EIATTR_MIN_STACK_SIZE
            elif sh.name == ".nv.info" and param == 0x2f: self.regs_usage = struct.unpack_from("II", data)[1] # EIATTR_REGCOUNT

      # Apply relocs
      for apply_image_offset, rel_sym_offset, typ, _ in relocs: # pylint: disable=possibly-used-before-assignment
        # These types are CUDA-specific, applying them here
        if typ == 2: image[apply_image_offset:apply_image_offset+8] = struct.pack('<Q', self.lib_gpu.va_addr + rel_sym_offset) # R_CUDA_64
        elif typ == 0x38: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) & 0xffffffff)
        elif typ == 0x39: image[apply_image_offset+4:apply_image_offset+8] = struct.pack('<I', (self.lib_gpu.va_addr + rel_sym_offset) >> 32)
        else: raise RuntimeError(f"unknown NV reloc {typ}")

      # Minimum cbuf_0 size for driver params: Blackwell needs index 223 (224 entries), older GPUs need index 11 (12 entries)
      min_cbuf0_entries = 224 if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A else 12
      self.cbuf_0 = [0] * max(cbuf0_size // 4, min_cbuf0_entries)

    # Ensure device has enough local memory to run the program
    self.dev._ensure_has_local_memory(self.lcmem_usage)
    self.dev.allocator._copyin(self.lib_gpu, image)
    self.dev.synchronize()

    if dev.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A:
      if not NAK: self.cbuf_0[188:192], self.cbuf_0[223] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window)], 0xfffdc0
      qmd = {'qmd_major_version':5, 'qmd_type':nv_gpu.NVCEC0_QMDV05_00_QMD_TYPE_GRID_CTA, 'program_address_upper_shifted4':hi32(prog_addr>>4),
        'program_address_lower_shifted4':lo32(prog_addr>>4), 'register_count':self.regs_usage, 'shared_memory_size_shifted7':self.shmem_usage>>7,
        'shader_local_memory_high_size_shifted4':self.lcmem_usage>>4 if NAK else self.dev.slm_per_thread>>4}
    else:
      if not NAK: self.cbuf_0[6:12] = [*data64_le(self.dev.shared_mem_window), *data64_le(self.dev.local_mem_window), *data64_le(0xfffdc0)]
      qmd = {'qmd_major_version':3, 'sm_global_caching_enable':1, 'program_address_upper':hi32(prog_addr), 'program_address_lower':lo32(prog_addr),
        'shared_memory_size':self.shmem_usage, 'register_count_v':self.regs_usage,
        **({'shader_local_memory_low_size':self.lcmem_usage} if NAK else {'shader_local_memory_high_size':self.dev.slm_per_thread})}

    smem_cfg = min(shmem_conf * 1024 for shmem_conf in [32, 64, 100] if shmem_conf * 1024 >= self.shmem_usage) // 4096 + 1

    self.qmd:QMD = QMD(dev, **qmd, qmd_group_id=0x3f, invalidate_texture_header_cache=1, invalidate_texture_sampler_cache=1,
      invalidate_texture_data_cache=1, invalidate_shader_data_cache=1, api_visible_call_limit=1, sampler_index=1, barrier_count=1,
      cwd_membar_type=nv_gpu.NVC6C0_QMDV03_00_CWD_MEMBAR_TYPE_L1_SYSMEMBAR, constant_buffer_invalidate_0=1, min_sm_config_shared_mem_size=smem_cfg,
      target_sm_config_shared_mem_size=smem_cfg, max_sm_config_shared_mem_size=0x1a, program_prefetch_size=min(prog_sz>>8, 0x1ff),
      sass_version=dev.sass_version, program_prefetch_addr_upper_shifted=prog_addr>>40, program_prefetch_addr_lower_shifted=prog_addr>>8)

    for i,(addr,sz) in self.constbufs.items():
      self.qmd.set_constant_buf_addr(i, addr)
      self.qmd.write(**{f'constant_buffer_size_shifted4_{i}': sz, f'constant_buffer_valid_{i}': 1})

    # Registers allocation granularity per warp is 256, warp allocation granularity is 4. Register file size is 65536.
    self.max_threads = ((65536 // round_up(max(1, self.regs_usage) * 32, 256)) // 4) * 4 * 32

    # NV's kernargs is constbuffer, then arguments to the kernel follows. Kernargs also appends QMD at the end of the kernel.
    super().__init__(NVArgsState, self.dev, self.name, kernargs_alloc_size=round_up(self.constbufs[0][1], 1 << 8) + (8 << 8))
    weakref.finalize(self, self._fini, self.dev, self.lib_gpu, buf_spec)

  def _parse_elf_info(self, sh, start_off=0):
    while start_off < sh.header.sh_size:
      typ, param, sz = struct.unpack_from("BBH", sh.content, start_off)
      yield typ, param, sh.content[start_off+4:start_off+sz+4] if typ == 0x4 else sz
      start_off += (sz if typ == 0x4 else 0) + 4

  def __call__(self, *bufs, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int|None, ...]=(), wait=False):
    if prod(local_size) > 1024 or self.max_threads < prod(local_size) or self.lcmem_usage > cast(NVDevice, self.dev).slm_per_thread:
      raise RuntimeError(f"Too many resources requested for launch, {prod(local_size)=}, {self.max_threads=}")
    if any(cur > mx for cur,mx in zip(global_size, [2147483647, 65535, 65535])) or any(cur > mx for cur,mx in zip(local_size, [1024, 1024, 64])):
      raise RuntimeError(f"Invalid global/local dims {global_size=}, {local_size=}")
    res = super().__call__(*bufs, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
    if self.dev.pma_enabled:
      self.dev.synchronize()
      if pma_blob:=self.dev._prof_readback():
        Compiled.profile_events += [ProfilePMAEvent(self.dev.device, self.name, pma_blob, self.dev.prof_exec_counter)]
    return res

class NVAllocator(HCQAllocator['NVDevice']):
  def _alloc(self, size:int, options:BufferSpec) -> HCQBuffer:
    uncached = options.uncached
    return self.dev.iface.alloc(size, cpu_access=options.cpu_access, host=options.host, uncached=uncached)

  def _do_free(self, opaque:HCQBuffer, options:BufferSpec): self.dev.iface.free(opaque)

  def _map(self, buf:HCQBuffer): return self.dev.iface.map(buf._base if buf._base is not None else buf)

  def _encode_decode(self, bufout:HCQBuffer, bufin:HCQBuffer, desc_buf:HCQBuffer, hist:list[HCQBuffer], shape:tuple[int,...], frame_pos:int):
    assert all(h.va_addr % 0x100 == 0 for h in hist + [bufin, bufout, desc_buf]), "all buffers must be 0x100 aligned"

    h, w = ((2 * shape[0]) // 3 if shape[0] % 3 == 0 else (2 * shape[0] - 1) // 3), shape[1]
    self.dev._ensure_has_vid_hw(w, h)

    q = NVVideoQueue().wait(self.dev.timeline_signal, self.dev.timeline_value - 1)
    with hcq_profile(self.dev, queue=q, desc="HEVC Decode", enabled=PROFILE, dev_suff="NVDEC"):
      q.decode_hevc_chunk(desc_buf, bufin, bufout, frame_pos, hist, [(frame_pos-x) % (len(hist) + 1) for x in range(len(hist), 0, -1)],
                          round_up(w, 64)*round_up(h, 64), self.dev.vid_coloc_buf, self.dev.vid_filter_buf, self.dev.intra_top_off,
                          self.dev.intra_unk_off, self.dev.vid_stat_buf)
    q.signal(self.dev.timeline_signal, self.dev.next_timeline()).submit(self.dev)

@dataclass
class GPFifo:
  ring: MMIOInterface
  gpput: MMIOInterface
  entries_count: int
  token: int
  put_value: int = 0

class NVKIface:
  root = None
  fd_ctl: FileIOInterface
  fd_uvm: FileIOInterface
  gpus_info: list|ctypes.Array = []

  # TODO: Need a proper allocator for va addresses
  # 0x1000000000 - 0x2000000000, reserved for system/cpu mappings
  # VA space is 48bits.
  low_uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=0x1000000000, base=0x8000000000 if OSX else 0x1000000000, wrap=False)
  uvm_vaddr_allocator: BumpAllocator = BumpAllocator(size=(1 << 48) - 1, base=low_uvm_vaddr_allocator.base + low_uvm_vaddr_allocator.size, wrap=False)
  host_object_enumerator: int = 0x1000

  def __init__(self, dev, device_id):
    if NVKIface.root is None:
      global nv_gpu

      NVKIface.fd_ctl = FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.fd_uvm = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      self.fd_uvm_2 = FileIOInterface("/dev/nvidia-uvm", os.O_RDWR | os.O_CLOEXEC)
      NVKIface.root = self.rm_alloc(0, nv_gpu.NV01_ROOT_CLIENT, None, root=0)

      drvver = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION_V2, nv_gpu.NV0000_CTRL_SYSTEM_GET_BUILD_VERSION_V2_PARAMS())
      if int(drvver.driverVersionBuffer.decode().split('.')[0], 10) >= 580: nv_gpu = nv_580

      self.uvm(nv_gpu.UVM_INITIALIZE, nv_gpu.UVM_INITIALIZE_PARAMS())

      # this error is okay, CUDA hits it too
      with contextlib.suppress(RuntimeError): self.uvm(nv_gpu.UVM_MM_INITIALIZE, nv_gpu.UVM_MM_INITIALIZE_PARAMS(uvmFd=self.fd_uvm.fd), self.fd_uvm_2)

      nv_iowr(NVKIface.fd_ctl, nv_gpu.NV_ESC_CARD_INFO, gpus_info:=(nv_gpu.nv_ioctl_card_info_t*64)())
      NVKIface.gpus_info = hcq_filter_visible_devices(gpus_info)

    self.dev, self.device_id = dev, device_id
    if self.device_id >= len(NVKIface.gpus_info) or not NVKIface.gpus_info[self.device_id].valid:
      raise RuntimeError(f"No device found for {device_id}. Requesting more devices than the system has?")

    self.fd_dev = self._new_gpu_fd()
    self.gpu_info = self.rm_control(self.root, nv_gpu.NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2,
      nv_gpu.NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS(gpuId=NVKIface.gpus_info[self.device_id].gpu_id))
    self.gpu_minor = NVKIface.gpus_info[self.device_id].minor_number
    self.gpu_instance = self.gpu_info.deviceInstance

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_ALLOC, made:=nv_gpu.NVOS21_PARAMETERS(hRoot=root if root is not None else self.root,
      hObjectParent=parent, hClass=clss, pAllocParms=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status == nv_gpu.NV_ERR_NO_MEMORY: raise MemoryError(f"rm_alloc returned {get_error_str(made.status)}")
    if made.status != 0: raise RuntimeError(f"rm_alloc returned {get_error_str(made.status)}")
    return made.hObjectNew

  def rm_control(self, obj, cmd, params=None, **kwargs):
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_CONTROL, made:=nv_gpu.NVOS54_PARAMETERS(hClient=self.root, hObject=obj, cmd=cmd,
      paramsSize=ctypes.sizeof(params) if params is not None else 0,
      params=ctypes.cast(ctypes.byref(params), ctypes.c_void_p) if params is not None else None))
    if made.status != 0: raise RuntimeError(f"rm_control returned {get_error_str(made.status)}")
    return params

  def uvm(self, cmd, params, fd=None):
    nv_iowr(fd or self.fd_uvm, None, params, cmd=cmd)
    if params.rmStatus != 0: raise RuntimeError(f"uvm returned {get_error_str(params.rmStatus)}")

  def setup_usermode(self):
    clsnum = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(numClasses=0))
    clsinfo = self.rm_control(self.dev.nvdevice, nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST, nv_gpu.NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS(
      numClasses=clsnum.numClasses, classList=mv_address(classlist:=memoryview(bytearray(clsnum.numClasses * 4)).cast('I'))))
    self.nvclasses = {classlist[i] for i in range(clsinfo.numClasses)}
    self.usermode_class:int = next(c for c in [nv_gpu.HOPPER_USERMODE_A, nv_gpu.TURING_USERMODE_A] if c in self.nvclasses)
    self.gpfifo_class:int = next(c for c in [nv_gpu.BLACKWELL_CHANNEL_GPFIFO_A, nv_gpu.AMPERE_CHANNEL_GPFIFO_A] if c in self.nvclasses)
    self.compute_class:int = next(c for c in [nv_gpu.BLACKWELL_COMPUTE_B, nv_gpu.ADA_COMPUTE_A, nv_gpu.AMPERE_COMPUTE_B] if c in self.nvclasses)
    self.dma_class:int = next(c for c in [nv_gpu.BLACKWELL_DMA_COPY_B, nv_gpu.AMPERE_DMA_COPY_B] if c in self.nvclasses)
    self.viddec_class:int|None = next((c for c in [nv_gpu.NVCFB0_VIDEO_DECODER, nv_gpu.NVC9B0_VIDEO_DECODER] if c in self.nvclasses), None)

    usermode = self.rm_alloc(self.dev.subdevice, self.usermode_class)
    return usermode, MMIOInterface(self._gpu_map_to_cpu(usermode, mmio_sz:=0x10000), mmio_sz, fmt='I')

  def setup_vm(self, vaspace):
    self.rm_control(self.dev.subdevice, nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO, raw_uuid:=nv_gpu.NV2080_CTRL_GPU_GET_GID_INFO_PARAMS(
      flags=nv_gpu.NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY, length=16))
    self.gpu_uuid = nv_gpu.struct_nv_uuid(uuid=(ctypes.c_ubyte*16)(*[raw_uuid.data[i] for i in range(16)]))

    self.uvm(nv_gpu.UVM_REGISTER_GPU, nv_gpu.UVM_REGISTER_GPU_PARAMS(rmCtrlFd=-1, gpu_uuid=self.gpu_uuid))
    self.uvm(nv_gpu.UVM_REGISTER_GPU_VASPACE, nv_gpu.UVM_REGISTER_GPU_VASPACE_PARAMS(
      gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hVaSpace=vaspace))

    for dev in cast(list[NVDevice], [d for pg in HCQCompiled.peer_groups.values() for d in pg if isinstance(d, NVDevice) and not d.is_nvd()]):
      try: self.uvm(nv_gpu.UVM_ENABLE_PEER_ACCESS, nv_gpu.UVM_ENABLE_PEER_ACCESS_PARAMS(gpuUuidA=self.gpu_uuid, gpuUuidB=dev.iface.gpu_uuid))
      except RuntimeError as e: raise RuntimeError(f"{e}. Make sure GPUs #{self.gpu_minor} & #{dev.iface.gpu_minor} have P2P enabled.") from e

  def setup_gpfifo_vm(self, gpfifo):
    self.uvm(nv_gpu.UVM_REGISTER_CHANNEL, nv_gpu.UVM_REGISTER_CHANNEL_PARAMS(gpuUuid=self.gpu_uuid, rmCtrlFd=self.fd_ctl.fd, hClient=self.root,
      hChannel=gpfifo, base=self._alloc_gpu_vaddr(0x4000000, force_low=True), length=0x4000000))

  def _new_gpu_fd(self):
    fd_dev = FileIOInterface(f"/dev/nvidia{NVKIface.gpus_info[self.device_id].minor_number}", os.O_RDWR | os.O_CLOEXEC)
    nv_iowr(fd_dev, nv_gpu.NV_ESC_REGISTER_FD, nv_gpu.nv_ioctl_register_fd_t(ctl_fd=self.fd_ctl.fd))
    return fd_dev

  def _gpu_map_to_cpu(self, memory_handle, size, target=None, flags=0, system=False):
    fd_dev = self._new_gpu_fd() if not system else FileIOInterface("/dev/nvidiactl", os.O_RDWR | os.O_CLOEXEC)
    made = nv_gpu.nv_ioctl_nvos33_parameters_with_fd(fd=fd_dev.fd,
      params=nv_gpu.NVOS33_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hMemory=memory_handle, length=size, flags=flags))
    nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY, made)
    if made.params.status != 0: raise RuntimeError(f"_gpu_map_to_cpu returned {get_error_str(made.params.status)}")
    return fd_dev.mmap(target, size, mmap.PROT_READ|mmap.PROT_WRITE, mmap.MAP_SHARED | (MAP_FIXED if target is not None else 0), 0)

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, map_flags=0, cpu_addr=None, **kwargs) -> HCQBuffer:
    # Uncached memory is "system". Use huge pages only for gpu memory.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (mmap.PAGESIZE if MOCKGPU else 4 << 10))
    size = round_up(size, page_size)
    va_addr = self._alloc_gpu_vaddr(size, alignment=page_size, force_low=cpu_access) if (alloced:=cpu_addr is None) else cpu_addr

    if host:
      if alloced: va_addr = FileIOInterface.anon_mmap(va_addr, size, mmap.PROT_READ|mmap.PROT_WRITE, MAP_FIXED|mmap.MAP_SHARED|mmap.MAP_ANONYMOUS, 0)

      flags = (nv_gpu.NVOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS << 4) | (nv_gpu.NVOS02_FLAGS_COHERENCY_CACHED << 12) \
            | (nv_gpu.NVOS02_FLAGS_MAPPING_NO_MAP << 30)

      NVKIface.host_object_enumerator += 1
      made = nv_gpu.nv_ioctl_nvos02_parameters_with_fd(params=nv_gpu.NVOS02_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, flags=flags,
        hObjectNew=NVKIface.host_object_enumerator, hClass=nv_gpu.NV01_MEMORY_SYSTEM_OS_DESCRIPTOR, pMemory=va_addr, limit=size-1), fd=-1)
      nv_iowr(self.fd_dev, nv_gpu.NV_ESC_RM_ALLOC_MEMORY, made)

      if made.params.status != 0: raise RuntimeError(f"host alloc returned {get_error_str(made.params.status)}")
      mem_handle = made.params.hObjectNew
    else:
      attr = ((nv_gpu.NVOS32_ATTR_PHYSICALITY_CONTIGUOUS if contiguous else nv_gpu.NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS) << 27) \
          | (nv_gpu.NVOS32_ATTR_PAGE_SIZE_HUGE if page_size > 0x1000 else 0) << 23 | ((nv_gpu.NVOS32_ATTR_LOCATION_PCI if uncached else 0) << 25)

      attr2 = ((nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_NO if uncached else nv_gpu.NVOS32_ATTR2_GPU_CACHEABLE_YES) << 2) \
            | ((nv_gpu.NVOS32_ATTR2_PAGE_SIZE_HUGE_2MB if page_size > 0x1000 else 0) << 20) | nv_gpu.NVOS32_ATTR2_ZBC_PREFER_NO_ZBC \
            | ((nv_gpu.NVOS32_ATTR2_PROTECTION_USER_READ_ONLY << 22) if kwargs.get('read_only') else 0)

      fl = nv_gpu.NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | nv_gpu.NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED | nv_gpu.NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE \
         | nv_gpu.NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT | (nv_gpu.NVOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM if not uncached else 0)

      alloc_func = nv_gpu.NV1_MEMORY_SYSTEM if uncached else nv_gpu.NV1_MEMORY_USER
      alloc_params = nv_gpu.NV_MEMORY_ALLOCATION_PARAMS(owner=self.root, alignment=page_size, offset=0, limit=size-1, format=6, size=size,
        type=nv_gpu.NVOS32_TYPE_NOTIFIER if uncached else nv_gpu.NVOS32_TYPE_IMAGE, attr=attr, attr2=attr2, flags=fl)
      mem_handle = self.rm_alloc(self.dev.nvdevice, alloc_func, alloc_params)

      if cpu_access: va_addr = self._gpu_map_to_cpu(mem_handle, size, target=va_addr, flags=map_flags, system=uncached)

    return self._gpu_uvm_map(va_addr, size, mem_handle, has_cpu_mapping=cpu_access or host)

  def free(self, mem:HCQBuffer):
    if mem.meta.hMemory > NVKIface.host_object_enumerator: # not a host object, clear phys mem.
      made = nv_gpu.NVOS00_PARAMETERS(hRoot=self.root, hObjectParent=self.dev.nvdevice, hObjectOld=mem.meta.hMemory)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_FREE, made)
      if made.status != 0: raise RuntimeError(f"_gpu_free returned {get_error_str(made.status)}")

    self.uvm(nv_gpu.UVM_FREE, nv_gpu.UVM_FREE_PARAMS(base=int(mem.va_addr), length=mem.size))
    if mem.view is not None: FileIOInterface.munmap(int(mem.va_addr), mem.size)

  def _gpu_uvm_map(self, va_base, size, mem_handle, create_range=True, has_cpu_mapping=False) -> HCQBuffer:
    if create_range:
      self.uvm(nv_gpu.UVM_CREATE_EXTERNAL_RANGE, nv_gpu.UVM_CREATE_EXTERNAL_RANGE_PARAMS(base=va_base, length=size))
      made = nv_gpu.NVOS46_PARAMETERS(hClient=self.root, hDevice=self.dev.nvdevice, hDma=self.dev.virtmem, hMemory=mem_handle, length=size,
        flags=(nv_gpu.NVOS46_FLAGS_PAGE_SIZE_4KB<<8)|(nv_gpu.NVOS46_FLAGS_CACHE_SNOOP_ENABLE<<4)|(nv_gpu.NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE<<15),
        dmaOffset=va_base)
      nv_iowr(self.fd_ctl, nv_gpu.NV_ESC_RM_MAP_MEMORY_DMA, made)
      if made.status != 0: raise RuntimeError(f"nv_sys_alloc 1 returned {get_error_str(made.status)}")
      assert made.dmaOffset == va_base, f"made.dmaOffset != va_base {made.dmaOffset=} {va_base=}"

    attrs = (nv_gpu.UvmGpuMappingAttributes*256)(nv_gpu.UvmGpuMappingAttributes(gpuUuid=self.gpu_uuid, gpuMappingType=1))

    self.uvm(nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION, uvm_map:=nv_gpu.UVM_MAP_EXTERNAL_ALLOCATION_PARAMS(base=va_base, length=size,
      rmCtrlFd=self.fd_ctl.fd, hClient=self.root, hMemory=mem_handle, gpuAttributesCount=1, perGpuAttributes=attrs, mapped_gpu_ids=[self.gpu_uuid]))
    return HCQBuffer(va_base, size, meta=uvm_map, view=MMIOInterface(va_base, size, fmt='B') if has_cpu_mapping else None, owner=self.dev)

  def map(self, mem:HCQBuffer):
    if mem.owner is not None and mem.owner._is_cpu():
      if not any(x.device.startswith("NV") for x in mem.mapped_devs): return self.alloc(mem.size, host=True, cpu_addr=mem.va_addr)
      mem = mem.mappings[next(x for x in mem.mapped_devs if x.device.startswith("NV"))]
    self._gpu_uvm_map(mem.va_addr, mem.size, mem.meta.hMemory, create_range=False)

  def _alloc_gpu_vaddr(self, size, alignment=(4 << 10), force_low=False):
    return NVKIface.low_uvm_vaddr_allocator.alloc(size, alignment) if force_low else NVKIface.uvm_vaddr_allocator.alloc(size, alignment)

  def sleep(self, tm:int): pass

class PCIIface(PCIIfaceBase):
  gpus:ClassVar[list[str]] = []

  def __init__(self, dev, dev_id):
    # PCIIface's MAP_FIXED mmap will overwrite UVM allocations made by NVKIface, so don't try PCIIface if kernel driver was already used.
    if NVKIface.root is not None: raise RuntimeError("Cannot use PCIIface after NVKIface has been initialized (would corrupt UVM memory)")
    super().__init__(dev, dev_id, vendor=0x10de, devices=[(0xff00, [0x2200, 0x2400, 0x2500, 0x2600, 0x2700, 0x2800, 0x2b00, 0x2c00, 0x2d00, 0x2f00])],
      base_class=0x03, bars=[0, 1, 3], vram_bar=1, va_start=NVMemoryManager.va_allocator.base, va_size=NVMemoryManager.va_allocator.size)
    if not OSX: System.reserve_hugepages(64)

    self.pci_dev.write_config(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.dev_impl:NVDev = NVDev(self.pci_dev)
    self.root, self.gpu_instance = 0xc1000000, 0
    self.rm_alloc(0, nv_gpu.NV01_ROOT, nv_gpu.NV0000_ALLOC_PARAMETERS())

    # Setup classes for the GPU
    self.gpfifo_class, self.compute_class, self.dma_class = (gsp:=self.dev_impl.gsp).gpfifo_class, gsp.compute_class, gsp.dma_class
    self.viddec_class = None

  def alloc(self, size:int, host=False, uncached=False, cpu_access=False, contiguous=False, **kwargs) -> HCQBuffer:
    # Force use of huge pages for large allocations. NVDev will attempt to use huge pages in any case,
    # but if the size is not aligned, the tail will be allocated with 4KB pages, increasing TLB pressure.
    page_size = mmap.PAGESIZE if uncached or host else ((2 << 20) if size >= (8 << 20) else (4 << 10))
    return super().alloc(round_up(size, page_size), host=host, uncached=uncached, cpu_access=cpu_access, contiguous=contiguous, **kwargs)

  def setup_usermode(self): return 0xce000000, self.pci_dev.map_bar(bar=0, fmt='I', off=0xbb0000, size=0x10000)
  def setup_vm(self, vaspace): pass
  def setup_gpfifo_vm(self, gpfifo): pass

  def rm_alloc(self, parent, clss, params=None, root=None) -> int: return self.dev_impl.gsp.rpc_rm_alloc(parent, clss, params, self.root)
  def rm_control(self, obj, cmd, params=None, **kwargs): return self.dev_impl.gsp.rpc_rm_control(obj, cmd, params, self.root, **kwargs)

  def device_fini(self): self.dev_impl.fini()

  def sleep(self, timeout):
    for _ in self.dev_impl.gsp.stat_q.read_resp(): pass
    if self.dev_impl.is_err_state: raise RuntimeError("Device fault detected")

# ============================================================================
# TegraIface — nvgpu/nvmap backend for Jetson Orin (JetPack 6)
# ============================================================================

# Linux ioctl direction bits (aarch64)
_IOC_NONE  = 0; _IOC_WRITE = 1; _IOC_READ  = 2
def _tegra_IOC(d, t, nr, size): return (d << 30) | (size << 16) | (ord(t) << 8) | nr
def _tegra_IO(t, nr): return _tegra_IOC(_IOC_NONE, t, nr, 0)
def _tegra_IOR(t, nr, sz): return _tegra_IOC(_IOC_READ, t, nr, sz)
def _tegra_IOW(t, nr, sz): return _tegra_IOC(_IOC_WRITE, t, nr, sz)
def _tegra_IOWR(t, nr, sz): return _tegra_IOC(_IOC_READ | _IOC_WRITE, t, nr, sz)

# ctypes structs for nvgpu/nvmap ioctls
class _nvgpu_gpu_characteristics(ctypes.Structure):
  _fields_ = [
    ("arch", ctypes.c_uint32), ("impl", ctypes.c_uint32), ("rev", ctypes.c_uint32), ("num_gpc", ctypes.c_uint32),
    ("numa_domain_id", ctypes.c_int32), ("_pad0", ctypes.c_uint32),
    ("L2_cache_size", ctypes.c_uint64), ("on_board_video_memory_size", ctypes.c_uint64),
    ("num_tpc_per_gpc", ctypes.c_uint32), ("bus_type", ctypes.c_uint32), ("big_page_size", ctypes.c_uint32),
    ("compression_page_size", ctypes.c_uint32), ("pde_coverage_bit_count", ctypes.c_uint32),
    ("available_big_page_sizes", ctypes.c_uint32),
    ("flags", ctypes.c_uint64),
    ("twod_class", ctypes.c_uint32), ("threed_class", ctypes.c_uint32), ("compute_class", ctypes.c_uint32),
    ("gpfifo_class", ctypes.c_uint32), ("inline_to_memory_class", ctypes.c_uint32), ("dma_copy_class", ctypes.c_uint32),
    ("gpc_mask", ctypes.c_uint32), ("sm_arch_sm_version", ctypes.c_uint32), ("sm_arch_spa_version", ctypes.c_uint32),
    ("sm_arch_warp_count", ctypes.c_uint32),
    ("gpu_ioctl_nr_last", ctypes.c_int16), ("tsg_ioctl_nr_last", ctypes.c_int16), ("dbg_gpu_ioctl_nr_last", ctypes.c_int16),
    ("ioctl_channel_nr_last", ctypes.c_int16), ("as_ioctl_nr_last", ctypes.c_int16),
    ("gpu_va_bit_count", ctypes.c_uint8), ("reserved", ctypes.c_uint8),
    ("max_fbps_count", ctypes.c_uint32), ("fbp_en_mask", ctypes.c_uint32), ("emc_en_mask", ctypes.c_uint32),
    ("max_ltc_per_fbp", ctypes.c_uint32), ("max_lts_per_ltc", ctypes.c_uint32), ("max_tex_per_tpc", ctypes.c_uint32),
    ("max_gpc_count", ctypes.c_uint32),
    ("rop_l2_en_mask_DEPRECATED", ctypes.c_uint32 * 2),
    ("chipname", ctypes.c_uint8 * 8),
    ("gr_compbit_store_base_hw", ctypes.c_uint64),
    ("gr_gobs_per_comptagline_per_slice", ctypes.c_uint32), ("num_ltc", ctypes.c_uint32),
    ("lts_per_ltc", ctypes.c_uint32), ("cbc_cache_line_size", ctypes.c_uint32),
    ("cbc_comptags_per_line", ctypes.c_uint32), ("map_buffer_batch_limit", ctypes.c_uint32),
    ("max_freq", ctypes.c_uint64),
    ("graphics_preemption_mode_flags", ctypes.c_uint32), ("compute_preemption_mode_flags", ctypes.c_uint32),
    ("default_graphics_preempt_mode", ctypes.c_uint32), ("default_compute_preempt_mode", ctypes.c_uint32),
    ("local_video_memory_size", ctypes.c_uint64),
    ("pci_vendor_id", ctypes.c_uint16), ("pci_device_id", ctypes.c_uint16), ("pci_subsystem_vendor_id", ctypes.c_uint16),
    ("pci_subsystem_device_id", ctypes.c_uint16), ("pci_class", ctypes.c_uint16), ("pci_revision", ctypes.c_uint8),
    ("vbios_oem_version", ctypes.c_uint8), ("vbios_version", ctypes.c_uint32),
    ("reg_ops_limit", ctypes.c_uint32), ("reserved1", ctypes.c_uint32),
    ("event_ioctl_nr_last", ctypes.c_int16), ("pad", ctypes.c_uint16), ("max_css_buffer_size", ctypes.c_uint32),
    ("ctxsw_ioctl_nr_last", ctypes.c_int16), ("prof_ioctl_nr_last", ctypes.c_int16),
    ("nvs_ioctl_nr_last", ctypes.c_int16), ("reserved2", ctypes.c_uint8 * 2),
    ("max_ctxsw_ring_buffer_size", ctypes.c_uint32), ("reserved3", ctypes.c_uint32),
    ("per_device_identifier", ctypes.c_uint64),
    ("num_ppc_per_gpc", ctypes.c_uint32), ("max_veid_count_per_tsg", ctypes.c_uint32),
    ("num_sub_partition_per_fbpa", ctypes.c_uint32), ("gpu_instance_id", ctypes.c_uint32),
    ("gr_instance_id", ctypes.c_uint32), ("max_gpfifo_entries", ctypes.c_uint32),
    ("max_dbg_tsg_timeslice", ctypes.c_uint32), ("reserved5", ctypes.c_uint32),
    ("device_instance_id", ctypes.c_uint64),
  ]

class _nvgpu_gpu_get_characteristics(ctypes.Structure):
  _fields_ = [("gpu_characteristics_buf_size", ctypes.c_uint64), ("gpu_characteristics_buf_addr", ctypes.c_uint64)]

class _nvmap_create_handle(ctypes.Structure):
  _fields_ = [("size", ctypes.c_uint32), ("handle", ctypes.c_uint32)]

class _nvmap_alloc_handle(ctypes.Structure):
  _fields_ = [("handle", ctypes.c_uint32), ("heap_mask", ctypes.c_uint32), ("flags", ctypes.c_uint32),
              ("align", ctypes.c_uint32), ("numa_nid", ctypes.c_int32)]

class _nvgpu_alloc_as_args(ctypes.Structure):
  _fields_ = [("big_page_size", ctypes.c_uint32), ("as_fd", ctypes.c_int32), ("flags", ctypes.c_uint32),
              ("reserved", ctypes.c_uint32), ("va_range_start", ctypes.c_uint64), ("va_range_end", ctypes.c_uint64),
              ("va_range_split", ctypes.c_uint64), ("padding", ctypes.c_uint32 * 6)]

class _nvgpu_as_bind_channel_args(ctypes.Structure):
  _fields_ = [("channel_fd", ctypes.c_uint32)]

class _nvgpu_as_map_buffer_ex_args(ctypes.Structure):
  _fields_ = [("flags", ctypes.c_uint32), ("compr_kind", ctypes.c_int16), ("incompr_kind", ctypes.c_int16),
              ("dmabuf_fd", ctypes.c_uint32), ("page_size", ctypes.c_uint32), ("buffer_offset", ctypes.c_uint64),
              ("mapping_size", ctypes.c_uint64), ("offset", ctypes.c_uint64)]

class _nvgpu_as_unmap_buffer_args(ctypes.Structure):
  _fields_ = [("offset", ctypes.c_uint64)]

class _nvgpu_gpu_open_tsg_args(ctypes.Structure):
  _fields_ = [("tsg_fd", ctypes.c_int32), ("flags", ctypes.c_uint32), ("token", ctypes.c_uint32),
              ("reserved", ctypes.c_uint32), ("subctx_id", ctypes.c_uint32), ("_pad", ctypes.c_uint32)]

class _nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
  _fields_ = [("channel_fd", ctypes.c_int32), ("subcontext_id", ctypes.c_uint32), ("reserved", ctypes.c_uint8 * 16)]

class _nvgpu_tsg_create_subcontext_args(ctypes.Structure):
  _fields_ = [("type", ctypes.c_uint32), ("as_fd", ctypes.c_int32), ("veid", ctypes.c_uint32), ("reserved", ctypes.c_uint32)]

class _nvgpu_gpu_open_channel_args(ctypes.Structure):
  _fields_ = [("channel_fd", ctypes.c_int32)]

class _nvgpu_alloc_obj_ctx_args(ctypes.Structure):
  _fields_ = [("class_num", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("obj_id", ctypes.c_uint64)]

class _nvgpu_channel_setup_bind_args(ctypes.Structure):
  _fields_ = [("num_gpfifo_entries", ctypes.c_uint32), ("num_inflight_jobs", ctypes.c_uint32),
              ("flags", ctypes.c_uint32), ("userd_dmabuf_fd", ctypes.c_int32), ("gpfifo_dmabuf_fd", ctypes.c_int32),
              ("work_submit_token", ctypes.c_uint32), ("userd_dmabuf_offset", ctypes.c_uint64),
              ("gpfifo_dmabuf_offset", ctypes.c_uint64), ("gpfifo_gpu_va", ctypes.c_uint64),
              ("userd_gpu_va", ctypes.c_uint64), ("usermode_mmio_gpu_va", ctypes.c_uint64),
              ("reserved", ctypes.c_uint32 * 9)]

class _nvgpu_channel_wdt_args(ctypes.Structure):
  _fields_ = [("wdt_status", ctypes.c_uint32), ("timeout_ms", ctypes.c_uint32)]

class _nvgpu_get_user_syncpoint_args(ctypes.Structure):
  _fields_ = [("gpu_va", ctypes.c_uint64), ("syncpoint_id", ctypes.c_uint32), ("syncpoint_max", ctypes.c_uint32)]

# --- nvgpu/nvmap ioctl codes ---
_NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _tegra_IOWR('G', 5, ctypes.sizeof(_nvgpu_gpu_get_characteristics))
_NVGPU_GPU_IOCTL_ALLOC_AS = _tegra_IOWR('G', 8, ctypes.sizeof(_nvgpu_alloc_as_args))
_NVGPU_GPU_IOCTL_OPEN_TSG = _tegra_IOWR('G', 9, ctypes.sizeof(_nvgpu_gpu_open_tsg_args))
_NVGPU_GPU_IOCTL_OPEN_CHANNEL = _tegra_IOWR('G', 11, ctypes.sizeof(_nvgpu_gpu_open_channel_args))
_NVMAP_IOC_CREATE = _tegra_IOWR('N', 0, ctypes.sizeof(_nvmap_create_handle))
_NVMAP_IOC_ALLOC = _tegra_IOW('N', 3, ctypes.sizeof(_nvmap_alloc_handle))
_NVMAP_IOC_GET_FD = _tegra_IOWR('N', 15, ctypes.sizeof(_nvmap_create_handle))
_NVMAP_IOC_FREE = _tegra_IO('N', 4)
_NVGPU_AS_IOCTL_BIND_CHANNEL = _tegra_IOWR('A', 1, ctypes.sizeof(_nvgpu_as_bind_channel_args))
_NVGPU_AS_IOCTL_MAP_BUFFER_EX = _tegra_IOWR('A', 7, ctypes.sizeof(_nvgpu_as_map_buffer_ex_args))
_NVGPU_AS_IOCTL_UNMAP_BUFFER = _tegra_IOWR('A', 5, ctypes.sizeof(_nvgpu_as_unmap_buffer_args))
_NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _tegra_IOWR('T', 11, ctypes.sizeof(_nvgpu_tsg_bind_channel_ex_args))
_NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _tegra_IOWR('T', 18, ctypes.sizeof(_nvgpu_tsg_create_subcontext_args))
_NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _tegra_IOWR('H', 108, ctypes.sizeof(_nvgpu_alloc_obj_ctx_args))
_NVGPU_IOCTL_CHANNEL_SETUP_BIND = _tegra_IOWR('H', 128, ctypes.sizeof(_nvgpu_channel_setup_bind_args))
_NVGPU_IOCTL_CHANNEL_WDT = _tegra_IOW('H', 119, ctypes.sizeof(_nvgpu_channel_wdt_args))
_NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT = _tegra_IOR('H', 126, ctypes.sizeof(_nvgpu_get_user_syncpoint_args))
_NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER = _tegra_IOWR('H', 111, 24) # struct is 24 bytes

# nvmap heap/flag constants
_NVMAP_HEAP_IOVMM = (1 << 30)
_NVMAP_HANDLE_WRITE_COMBINE = 1
_NVMAP_HANDLE_INNER_CACHEABLE = 2
_NVMAP_TAG_TINYGRAD = 0x0900  # tag in bits [31:16] of flags — identifies subsystem to kernel (silences nvmap_alloc_handle WARNING)

# SETUP_BIND flags
_NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT = (1 << 3)
_NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC = (1 << 1)

def _tegra_ioctl(fd, ioc_code, buf):
  """Call an ioctl on a raw fd, raise on error."""
  ret = fcntl.ioctl(fd, ioc_code, buf)
  if ret < 0: raise OSError(f"tegra ioctl 0x{ioc_code:08x} failed with {ret}")
  return ret

@dataclass
class _TegraMem:
  """Metadata for a Tegra nvmap allocation."""
  handle: int
  dmabuf_fd: int
  gpu_va: int
  size: int
  cpu_addr: int = 0  # CPU mmap base address
  hMemory: int = 0  # compatibility with NVKIface's mem.meta.hMemory

class TegraIface:
  """nvgpu/nvmap backend for Jetson Orin (JetPack 6). Replaces NVKIface/PCIIface for Tegra SoC GPUs."""
  _inited: ClassVar[bool] = False
  _nvmap_fd: ClassVar[int] = -1
  _ctrl_fd: ClassVar[int] = -1
  _chars: ClassVar[_nvgpu_gpu_characteristics|None] = None

  def __init__(self, dev, device_id):
    if device_id != 0: raise RuntimeError("TegraIface only supports device 0 (single iGPU)")

    # Open device nodes (class-level, shared across instances)
    if not TegraIface._inited:
      if not os.path.exists("/dev/nvgpu/igpu0/ctrl"): raise FileNotFoundError("/dev/nvgpu/igpu0/ctrl")
      TegraIface._nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
      TegraIface._ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)

      # GET_CHARACTERISTICS
      chars = _nvgpu_gpu_characteristics()
      ctypes.memset(ctypes.addressof(chars), 0, ctypes.sizeof(chars))
      req = _nvgpu_gpu_get_characteristics()
      req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
      req.gpu_characteristics_buf_addr = ctypes.addressof(chars)
      _tegra_ioctl(TegraIface._ctrl_fd, _NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)
      TegraIface._chars = chars
      TegraIface._inited = True

    self.dev, self.device_id = dev, device_id
    chars = TegraIface._chars
    assert chars is not None

    # GPU classes from characteristics
    self.compute_class = chars.compute_class    # 0xc7c0 (AMPERE_COMPUTE_B)
    self.gpfifo_class = chars.gpfifo_class      # 0xc76f
    self.dma_class = chars.dma_copy_class       # 0xc7b5 (AMPERE_DMA_COPY_B)
    self.viddec_class = None                    # no video decode via nvgpu

    # GPU info for _query_gpu_info translation
    self._gpu_chars = chars
    self._sm_version = chars.sm_arch_sm_version  # 0x807 for SM 8.7

    # RM handle emulation: maps fake handles → state
    self._handle_counter = 0x10000
    self._handles: dict[int, dict] = {}

    # Address space state (created during rm_alloc of NV01_MEMORY_VIRTUAL)
    self._as_fd: int = -1

    # Channel/TSG state (created during rm_alloc of channel classes)
    self._tsg_fd: int = -1
    self._ch_fds: dict[int, int] = {}  # handle → channel fd
    self._subctx_veid: int = 0
    self._work_submit_tokens: dict[int, int] = {}  # channel handle → token
    self._gpfifo_setup: dict[int, dict] = {}  # channel handle → setup info

    # Root / device / subdevice fake handles
    self.root = self._next_handle()
    self.gpu_instance = 0

    # Allocations tracking
    self._allocs: list[_TegraMem] = []

  def _next_handle(self) -> int:
    self._handle_counter += 1
    return self._handle_counter

  def rm_alloc(self, parent, clss, params=None, root=None) -> int:
    """Translate RM object allocations to nvgpu equivalents."""
    handle = self._next_handle()

    if clss == nv_gpu.NV01_DEVICE_0:
      # No nvgpu equivalent — just a hierarchy container
      self._handles[handle] = {"type": "device"}
      return handle

    if clss == nv_gpu.NV20_SUBDEVICE_0:
      # No nvgpu equivalent — hierarchy container
      self._handles[handle] = {"type": "subdevice"}
      return handle

    if clss == nv_gpu.NV01_MEMORY_VIRTUAL:
      # Create address space via ALLOC_AS
      args = _nvgpu_alloc_as_args()
      PDE_SIZE = 1 << 21
      args.big_page_size = 0
      args.flags = 2  # UNIFIED_VA
      args.va_range_start = PDE_SIZE
      args.va_range_end = (1 << 40) - PDE_SIZE
      args.va_range_split = 0
      _tegra_ioctl(self._ctrl_fd, _NVGPU_GPU_IOCTL_ALLOC_AS, args)
      self._as_fd = args.as_fd
      self._handles[handle] = {"type": "virtmem", "as_fd": self._as_fd}
      return handle

    if clss == nv_gpu.FERMI_VASPACE_A:
      # Already have AS from NV01_MEMORY_VIRTUAL — NOP
      self._handles[handle] = {"type": "vaspace"}
      return handle

    if clss == nv_gpu.KEPLER_CHANNEL_GROUP_A:
      # Create TSG
      tsg_args = _nvgpu_gpu_open_tsg_args()
      tsg_args.flags = 0
      _tegra_ioctl(self._ctrl_fd, _NVGPU_GPU_IOCTL_OPEN_TSG, tsg_args)
      self._tsg_fd = tsg_args.tsg_fd
      self._handles[handle] = {"type": "tsg", "tsg_fd": self._tsg_fd}
      return handle

    if clss == nv_gpu.FERMI_CONTEXT_SHARE_A:
      # Create subcontext in TSG
      subctx = _nvgpu_tsg_create_subcontext_args()
      subctx.type = 1  # ASYNC for compute
      subctx.as_fd = self._as_fd
      _tegra_ioctl(self._tsg_fd, _NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, subctx)
      self._subctx_veid = subctx.veid
      self._handles[handle] = {"type": "ctxshare", "veid": subctx.veid}
      return handle

    if clss == self.gpfifo_class or clss == nv_gpu.AMPERE_CHANNEL_GPFIFO_A:
      # Full channel setup: OPEN_CHANNEL → AS_BIND → TSG_BIND → WDT → SETUP_BIND
      ch_args = _nvgpu_gpu_open_channel_args()
      ch_args.channel_fd = -1  # auto runlist
      _tegra_ioctl(self._ctrl_fd, _NVGPU_GPU_IOCTL_OPEN_CHANNEL, ch_args)
      ch_fd = ch_args.channel_fd

      # Bind channel to AS (must be before TSG bind)
      as_bind = _nvgpu_as_bind_channel_args()
      as_bind.channel_fd = ch_fd
      _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_BIND_CHANNEL, as_bind)

      # Bind channel to TSG
      tsg_bind = _nvgpu_tsg_bind_channel_ex_args()
      tsg_bind.channel_fd = ch_fd
      tsg_bind.subcontext_id = self._subctx_veid
      _tegra_ioctl(self._tsg_fd, _NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, tsg_bind)

      # Disable watchdog
      wdt = _nvgpu_channel_wdt_args()
      wdt.wdt_status = 1  # disable
      _tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_WDT, wdt)

      # Extract params from NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS
      gpfifo_va = 0
      userd_buf_offset = 0  # offset of userd region within the gpfifo_area dmabuf
      gpfifo_entries = 0x10000  # default
      gpfifo_buf_handle = 0

      if params is not None:
        gpfifo_va = params.gpFifoOffset
        gpfifo_entries = params.gpFifoEntries
        gpfifo_buf_handle = params.hObjectBuffer
        userd_buf_offset = params.userdOffset[0]  # entries*8 + offset — dmabuf-relative

      # Find the gpfifo_area's _TegraMem to get its cpu_addr for MAP_FIXED overlays
      gpfifo_area_mem = None
      for mem in self._allocs:
        if mem.hMemory == gpfifo_buf_handle:
          gpfifo_area_mem = mem
          break
      if gpfifo_area_mem is None:
        raise RuntimeError(f"TegraIface: can't find gpfifo_area alloc for handle {gpfifo_buf_handle}")

      # CRITICAL: nvgpu kernel constraints (linux-channel.c):
      #   1. gpfifo_dmabuf_offset MUST be 0 (non-zero returns -EINVAL)
      #   2. userd_dmabuf_offset MUST be 0 (non-zero returns -EINVAL)
      #   3. Kernel maps the ENTIRE dmabuf into GPU VA via nvgpu_gmmu_map
      #      → Large dmabufs (3MB) fragment VA space and crash ALLOC_OBJ_CTX
      # Solution: create per-channel SMALL dedicated nvmap buffers.

      gpfifo_ring_size = gpfifo_entries * 8  # e.g., 1024 * 8 = 8KB
      gpfifo_cpu_offset = gpfifo_va - gpfifo_area_mem.gpu_va  # offset within gpfifo_area for CPU addressing

      # --- Allocate dedicated gpfifo ring buffer (exactly entries*8 bytes) ---
      gcreate = _nvmap_create_handle()
      gcreate.size = gpfifo_ring_size
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_CREATE, gcreate)
      galloc = _nvmap_alloc_handle()
      galloc.handle = gcreate.handle
      galloc.heap_mask = _NVMAP_HEAP_IOVMM
      galloc.flags = (_NVMAP_TAG_TINYGRAD << 16) | _NVMAP_HANDLE_WRITE_COMBINE
      galloc.align = 4096
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_ALLOC, galloc)
      ggetfd = _nvmap_create_handle()
      ggetfd.handle = gcreate.handle
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_GET_FD, ggetfd)
      gpfifo_ring_dmabuf_fd = ggetfd.size  # dmabuf fd in .size field

      # --- Allocate dedicated 4KB userd buffer ---
      USERD_SIZE = 4096
      ucreate = _nvmap_create_handle()
      ucreate.size = USERD_SIZE
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_CREATE, ucreate)
      ualloc = _nvmap_alloc_handle()
      ualloc.handle = ucreate.handle
      ualloc.heap_mask = _NVMAP_HEAP_IOVMM
      ualloc.flags = (_NVMAP_TAG_TINYGRAD << 16) | _NVMAP_HANDLE_WRITE_COMBINE
      ualloc.align = 4096
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_ALLOC, ualloc)
      ugetfd = _nvmap_create_handle()
      ugetfd.handle = ucreate.handle
      _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_GET_FD, ugetfd)
      userd_dmabuf_fd = ugetfd.size  # dmabuf fd in .size field

      # SETUP_BIND with per-channel small buffers, both at offset 0
      setup = _nvgpu_channel_setup_bind_args()
      setup.num_gpfifo_entries = gpfifo_entries
      setup.num_inflight_jobs = 0
      setup.gpfifo_dmabuf_fd = gpfifo_ring_dmabuf_fd  # dedicated small buffer (NOT gpfifo_area)
      setup.gpfifo_dmabuf_offset = 0                  # MUST be 0 per kernel constraint
      setup.userd_dmabuf_fd = userd_dmabuf_fd          # dedicated 4KB buffer
      setup.userd_dmabuf_offset = 0                    # MUST be 0 per kernel constraint
      setup.flags = _NVGPU_SETUP_BIND_FLAGS_USERMODE_SUPPORT | _NVGPU_SETUP_BIND_FLAGS_DETERMINISTIC

      if getenv("DEBUG", 0) >= 1:
        print(f"TegraIface SETUP_BIND: ch_fd={ch_fd} entries={gpfifo_entries} ring_size={gpfifo_ring_size} "
              f"gpfifo_dmabuf_fd={gpfifo_ring_dmabuf_fd} userd_dmabuf_fd={userd_dmabuf_fd} flags=0x{setup.flags:x}")

      _tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_SETUP_BIND, setup)

      # MAP_FIXED overlay: replace gpfifo_area's pages with per-channel dmabuf pages.
      # This lets tinygrad's cpu_view() work unchanged — it reads from gpfifo_area's mmap,
      # but the physical backing is now the small per-channel dmabufs that the GPU also sees.
      import ctypes as ct
      libc_so = ct.CDLL("libc.so.6", use_errno=True)
      libc_so.mmap.restype = ct.c_void_p
      libc_so.mmap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long]

      # Overlay gpfifo ring at gpfifo_area_cpu + offset
      gpfifo_cpu_target = gpfifo_area_mem.cpu_addr + gpfifo_cpu_offset
      ring_addr = libc_so.mmap(ct.c_void_p(gpfifo_cpu_target), gpfifo_ring_size,
                               mmap.PROT_READ | mmap.PROT_WRITE,
                               mmap.MAP_SHARED | MAP_FIXED,
                               gpfifo_ring_dmabuf_fd, 0)
      if ring_addr is None or ring_addr == ct.c_void_p(-1).value or ring_addr == 0xffffffffffffffff:
        raise RuntimeError(f"TegraIface: gpfifo ring overlay mmap failed (errno={ct.get_errno()})")

      # Overlay userd at gpfifo_area_cpu + userd_buf_offset
      userd_cpu_target = gpfifo_area_mem.cpu_addr + userd_buf_offset
      overlay_addr = libc_so.mmap(ct.c_void_p(userd_cpu_target), USERD_SIZE,
                                  mmap.PROT_READ | mmap.PROT_WRITE,
                                  mmap.MAP_SHARED | MAP_FIXED,
                                  userd_dmabuf_fd, 0)
      if overlay_addr is None or overlay_addr == ct.c_void_p(-1).value or overlay_addr == 0xffffffffffffffff:
        raise RuntimeError(f"TegraIface: userd overlay mmap failed (errno={ct.get_errno()})")

      self._ch_fds[handle] = ch_fd
      self._work_submit_tokens[handle] = setup.work_submit_token
      self._gpfifo_setup[handle] = {
        "ch_fd": ch_fd, "token": setup.work_submit_token,
        "gpfifo_gpu_va": setup.gpfifo_gpu_va, "userd_gpu_va": setup.userd_gpu_va,
      }
      self._handles[handle] = {"type": "channel", "ch_fd": ch_fd, "token": setup.work_submit_token}
      return handle

    if clss == self.compute_class or clss == nv_gpu.AMPERE_COMPUTE_B:
      # Allocate compute class on channel
      ch_fd = self._ch_fds.get(parent, -1)
      if ch_fd == -1: raise RuntimeError(f"TegraIface: no channel fd for handle {parent}")
      obj = _nvgpu_alloc_obj_ctx_args()
      obj.class_num = self.compute_class
      _tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
      self._handles[handle] = {"type": "compute_obj"}
      return handle

    if clss == self.dma_class or clss == nv_gpu.AMPERE_DMA_COPY_B:
      # Allocate DMA copy class on channel
      ch_fd = self._ch_fds.get(parent, -1)
      if ch_fd == -1: raise RuntimeError(f"TegraIface: no channel fd for handle {parent}")
      obj = _nvgpu_alloc_obj_ctx_args()
      obj.class_num = self.dma_class
      _tegra_ioctl(ch_fd, _NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
      self._handles[handle] = {"type": "dma_obj"}
      return handle

    if clss == nv_gpu.GT200_DEBUGGER:
      # Skip debugger on Tegra — not needed
      self._handles[handle] = {"type": "debugger_stub"}
      return handle

    if clss == nv_gpu.NV01_ROOT_CLIENT or clss == nv_gpu.NV01_ROOT:
      self._handles[handle] = {"type": "root"}
      return handle

    if clss in (getattr(nv_gpu, 'NV1_MEMORY_SYSTEM', 0), getattr(nv_gpu, 'NV1_MEMORY_USER', 0),
                getattr(nv_gpu, 'NV01_MEMORY_SYSTEM_OS_DESCRIPTOR', 0)):
      # These are used for device memory allocation by NVKIface but we handle alloc() directly
      self._handles[handle] = {"type": "mem_alloc_stub"}
      return handle

    # For any other class, stub it out with a warning
    self._handles[handle] = {"type": f"stub_{clss:#x}"}
    return handle

  def rm_control(self, obj, cmd, params=None):
    """Translate RM control calls to nvgpu equivalents."""
    # NV2080_CTRL_CMD_PERF_BOOST — boost GPU frequency
    if cmd == nv_gpu.NV2080_CTRL_CMD_PERF_BOOST:
      # Try to set max GPU frequency via devfreq sysfs
      try:
        with open("/sys/class/devfreq/17000000.gpu/max_freq") as f: max_freq = f.read().strip()
        with open("/sys/class/devfreq/17000000.gpu/min_freq", "w") as f: f.write(max_freq)
      except (OSError, IOError): pass  # best-effort
      return params

    # NVA06C_CTRL_CMD_GPFIFO_SCHEDULE — NOP on Tegra (channels auto-schedule)
    if cmd == nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE:
      return params

    # NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN
    if cmd == nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN:
      # Return the saved token from SETUP_BIND for this channel handle
      token = self._work_submit_tokens.get(obj, 0)
      if params is not None: params.workSubmitToken = token
      return params

    # NV2080_CTRL_CMD_GR_GET_INFO — translate to GET_CHARACTERISTICS fields
    if cmd == nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO:
      chars = self._gpu_chars
      # Build a mapping from GR_INFO_INDEX to values from characteristics
      info_map = {
        getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS', None): chars.num_gpc,
        getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC', None): chars.num_tpc_per_gpc,
        getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC', None): 2,  # ga10b has 2 SMs per TPC
        getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM', None): chars.sm_arch_warp_count,
        getattr(nv_gpu, 'NV2080_CTRL_GR_INFO_INDEX_SM_VERSION', None): chars.sm_arch_sm_version,
      }
      info_map = {k: v for k, v in info_map.items() if k is not None}

      if params is not None:
        info_list_addr = params.grInfoList
        for i in range(params.grInfoListSize):
          info = nv_gpu.NV2080_CTRL_GR_INFO.from_address(info_list_addr + i * ctypes.sizeof(nv_gpu.NV2080_CTRL_GR_INFO))
          info.data = info_map.get(info.index, 0)
      return params

    # NV0080_CTRL_CMD_GPU_GET_CLASSLIST — return known classes
    if cmd == nv_gpu.NV0080_CTRL_CMD_GPU_GET_CLASSLIST:
      if params is not None:
        known_classes = [self.compute_class, self.gpfifo_class, self.dma_class, nv_gpu.TURING_USERMODE_A]
        if params.numClasses == 0:
          params.numClasses = len(known_classes)
        else:
          cl = to_mv(params.classList, params.numClasses * 4).cast('I')
          for i, c in enumerate(known_classes[:params.numClasses]): cl[i] = c
      return params

    # NV2080_CTRL_CMD_GPU_GET_GID_INFO — return synthetic UUID
    if cmd == nv_gpu.NV2080_CTRL_CMD_GPU_GET_GID_INFO:
      if params is not None:
        params.length = 16
        for i in range(16): params.data[i] = (0x4A + i) & 0xFF  # 'J' for Jetson + sequential
      return params

    # NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE — NOP on Tegra with IO_COHERENCE
    if cmd == getattr(nv_gpu, 'NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE', 0):
      return params

    # NVA06F_CTRL_CMD_BIND — NOP, channels already bound to engine during setup
    if cmd == getattr(nv_gpu, 'NVA06F_CTRL_CMD_BIND', 0):
      return params

    # NVA06F_CTRL_CMD_GPFIFO_SCHEDULE — NOP
    if cmd == getattr(nv_gpu, 'NVA06F_CTRL_CMD_GPFIFO_SCHEDULE', 0):
      return params

    # NV2080_CTRL_CMD_GR_GET_TPC_MASK — return TPC mask from characteristics
    if cmd == getattr(nv_gpu, 'NV2080_CTRL_CMD_GR_GET_TPC_MASK', 0):
      if params is not None: params.tpcMask = (1 << self._gpu_chars.num_tpc_per_gpc) - 1
      return params

    # For any unrecognized control cmd, return params as-is (NOP)
    return params

  def setup_usermode(self):
    """Return (handle, mmio_interface_for_doorbell).
    On Tegra, the doorbell is at offset 0x90 of the mmap'd ctrl fd."""
    import ctypes as ct
    libc_so = ct.CDLL("libc.so.6", use_errno=True)
    libc_so.mmap.restype = ct.c_void_p
    libc_so.mmap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long]
    addr = libc_so.mmap(None, 0x10000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, self._ctrl_fd, 0)
    if addr is None or addr == ct.c_void_p(-1).value or addr == 0xffffffffffffffff:
      raise RuntimeError(f"TegraIface: failed to mmap ctrl fd for doorbell (errno={ct.get_errno()})")

    return 0, MMIOInterface(addr, 0x10000, fmt='I')

  def setup_vm(self, vaspace):
    """On Tegra, AS is already set up — NOP."""
    pass

  def setup_gpfifo_vm(self, gpfifo):
    """On Tegra, channel is already bound to AS — NOP."""
    pass

  def alloc(self, size: int, host=False, uncached=False, cpu_access=False, contiguous=False,
            map_flags=0, cpu_addr=None, force_devmem=False, **kwargs) -> HCQBuffer:
    """Allocate GPU memory via nvmap: CREATE → ALLOC → GET_FD → MAP_BUFFER_EX → mmap."""
    page_size = mmap.PAGESIZE
    size = round_up(size, page_size)

    # Use write-combine for uncached/host buffers, inner-cacheable for device memory.
    # Note: cpu_access buffers (like kernargs_buf/QMD) also need write-combine for GPU coherence,
    # but that's handled separately via uncached=True flag at allocation site, not here.
    cache_flags = _NVMAP_HANDLE_WRITE_COMBINE if (uncached or host) else _NVMAP_HANDLE_INNER_CACHEABLE
    if kwargs.get('cpu_cached'): cache_flags = _NVMAP_HANDLE_INNER_CACHEABLE

    # Step 1: nvmap CREATE
    create = _nvmap_create_handle()
    create.size = size
    _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_CREATE, create)
    handle = create.handle

    # Step 2: nvmap ALLOC
    alloc_args = _nvmap_alloc_handle()
    alloc_args.handle = handle
    alloc_args.heap_mask = _NVMAP_HEAP_IOVMM
    alloc_args.flags = (_NVMAP_TAG_TINYGRAD << 16) | cache_flags
    alloc_args.align = page_size
    alloc_args.numa_nid = 0
    _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_ALLOC, alloc_args)

    # Step 3: GET_FD (dmabuf)
    get_fd = _nvmap_create_handle()
    get_fd.handle = handle
    _tegra_ioctl(self._nvmap_fd, _NVMAP_IOC_GET_FD, get_fd)
    dmabuf_fd = get_fd.size  # fd returned in size field

    # Step 4: MAP_BUFFER_EX — get GPU VA
    gpu_va = 0
    if self._as_fd >= 0:
      map_args = _nvgpu_as_map_buffer_ex_args()
      map_args.flags = 0  # let kernel pick address
      map_args.compr_kind = -1
      map_args.incompr_kind = 0
      map_args.dmabuf_fd = dmabuf_fd
      map_args.page_size = page_size
      map_args.buffer_offset = 0
      map_args.mapping_size = 0  # whole buffer
      map_args.offset = 0  # kernel picks
      _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_MAP_BUFFER_EX, map_args)
      gpu_va = map_args.offset

    # Step 5: mmap to CPU at the GPU VA address (unified VA)
    # On Tegra with unified memory, we mmap the dmabuf at the same address as the GPU VA using MAP_FIXED.
    # This means va_addr = GPU VA = CPU pointer, so code that uses va_addr as either works correctly.
    # This mirrors how desktop NVK/UVM unifies the address space, but we achieve it explicitly via MAP_FIXED.
    import ctypes as ct
    libc_so = ct.CDLL("libc.so.6", use_errno=True)
    libc_so.mmap.restype = ct.c_void_p
    libc_so.mmap.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_int, ct.c_int, ct.c_long]

    if gpu_va != 0:
      # Unified VA: mmap at the GPU VA address so va_addr works as both CPU pointer and GPU address
      addr = libc_so.mmap(ct.c_void_p(gpu_va), size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED | MAP_FIXED, dmabuf_fd, 0)
      if addr is None or addr == ct.c_void_p(-1).value or addr == 0xffffffffffffffff:
        # Fallback: mmap at kernel-chosen address if MAP_FIXED at GPU VA fails
        addr = libc_so.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, dmabuf_fd, 0)
        if addr is None or addr == ct.c_void_p(-1).value or addr == 0xffffffffffffffff:
          raise RuntimeError(f"TegraIface: mmap dmabuf_fd={dmabuf_fd} size={size} failed (errno={ct.get_errno()})")
    else:
      addr = libc_so.mmap(None, size, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, dmabuf_fd, 0)
      if addr is None or addr == ct.c_void_p(-1).value or addr == 0xffffffffffffffff:
        raise RuntimeError(f"TegraIface: mmap dmabuf_fd={dmabuf_fd} size={size} failed (errno={ct.get_errno()})")
    view = MMIOInterface(addr, size, fmt='B')

    # Assign a fake hMemory handle for RM-style tracking used by NVDevice
    fake_hmemory = self._next_handle()

    meta = _TegraMem(handle=handle, dmabuf_fd=dmabuf_fd, gpu_va=gpu_va, size=size, cpu_addr=addr, hMemory=fake_hmemory)
    self._allocs.append(meta)

    return HCQBuffer(va_addr=gpu_va, size=size, meta=meta, view=view, owner=self.dev)

  def free(self, mem: HCQBuffer):
    """Free a Tegra allocation."""
    meta = mem.meta
    if not isinstance(meta, _TegraMem): return

    # Unmap from GPU VA
    if meta.gpu_va and self._as_fd >= 0:
      try:
        unmap = _nvgpu_as_unmap_buffer_args()
        unmap.offset = meta.gpu_va
        _tegra_ioctl(self._as_fd, _NVGPU_AS_IOCTL_UNMAP_BUFFER, unmap)
      except OSError: pass

    # Unmap CPU view (use meta.cpu_addr which is the actual mmap return value)
    if meta.cpu_addr:
      try: FileIOInterface.munmap(meta.cpu_addr, meta.size)
      except Exception: pass

    # Close dmabuf fd
    if meta.dmabuf_fd >= 0:
      try: os.close(meta.dmabuf_fd)
      except OSError: pass

    # Free nvmap handle
    if meta.handle != 0:
      try:
        signed_handle = meta.handle if meta.handle < 0x80000000 else meta.handle - 0x100000000
        fcntl.ioctl(self._nvmap_fd, _NVMAP_IOC_FREE, signed_handle)
      except OSError: pass

    if meta in self._allocs: self._allocs.remove(meta)

  def map(self, mem: HCQBuffer):
    """Cross-device mapping. Tegra is single-GPU, so just identity-map."""
    pass

  def sleep(self, tm: int):
    """Sleep during long operations."""
    pass

class NVDevice(HCQCompiled[NVSignal]):
  def is_nvd(self) -> bool: return isinstance(self.iface, PCIIface)
  def is_tegra(self) -> bool: return isinstance(self.iface, TegraIface)

  def __init__(self, device:str=""):
    self.device_id = int(device.split(":")[1]) if ":" in device else 0
    self.iface = self._select_iface(TegraIface, NVKIface, PCIIface)

    device_params = nv_gpu.NV0080_ALLOC_PARAMETERS(deviceId=self.iface.gpu_instance, hClientShare=self.iface.root,
                                                   vaMode=nv_gpu.NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES)
    self.nvdevice = self.iface.rm_alloc(self.iface.root, nv_gpu.NV01_DEVICE_0, device_params)
    self.subdevice = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV20_SUBDEVICE_0, nv_gpu.NV2080_ALLOC_PARAMETERS())
    self.virtmem = self.iface.rm_alloc(self.nvdevice, nv_gpu.NV01_MEMORY_VIRTUAL, nv_gpu.NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS(limit=0x1ffffffffffff))
    self.usermode, self.gpu_mmio = self.iface.setup_usermode()

    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, nv_gpu.NV2080_CTRL_PERF_BOOST_PARAMS(duration=0xffffffff,
      flags=((nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_YES << 4) | (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CUDA_PRIORITY_HIGH << 6) | \
             (nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX))))

    vaspace_params = nv_gpu.NV_VASPACE_ALLOCATION_PARAMETERS(vaBase=0x1000, vaSize=0x1fffffb000000,
      flags=nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | nv_gpu.NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED)
    vaspace = self.iface.rm_alloc(self.nvdevice, nv_gpu.FERMI_VASPACE_A, vaspace_params)

    self.iface.setup_vm(vaspace)

    channel_params = nv_gpu.NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS(engineType=nv_gpu.NV2080_ENGINE_TYPE_GRAPHICS)
    self.channel_group = self.iface.rm_alloc(self.nvdevice, nv_gpu.KEPLER_CHANNEL_GROUP_A, channel_params)

    self.gpfifo_area = self.iface.alloc(0x10000 if self.is_tegra() else 0x300000, contiguous=True, cpu_access=True, force_devmem=True,
      map_flags=(nv_gpu.NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED<<23))

    ctxshare_params = nv_gpu.NV_CTXSHARE_ALLOCATION_PARAMETERS(hVASpace=vaspace, flags=nv_gpu.NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC)
    ctxshare = self.iface.rm_alloc(self.channel_group, nv_gpu.FERMI_CONTEXT_SHARE_A, ctxshare_params)

    # Tegra: use small per-channel buffers (1024 entries = 8KB ring) to avoid crashing ALLOC_OBJ_CTX.
    # The kernel maps the ENTIRE gpfifo dmabuf into GPU VA; large buffers fragment VA and crash GR context alloc.
    if self.is_tegra():
      tegra_entries = 1024
      self.compute_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0, entries=tegra_entries, compute=True)
      self.dma_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group,
                                            offset=tegra_entries * 8 + 0x1000, entries=tegra_entries, compute=False)
    else:
      self.compute_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0, entries=0x10000, compute=True)
      self.dma_gpfifo = self._new_gpu_fifo(self.gpfifo_area, ctxshare, self.channel_group, offset=0x100000, entries=0x10000, compute=False)
    self.iface.rm_control(self.channel_group, nv_gpu.NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    self.cmdq_page:HCQBuffer = self.iface.alloc(0x200000, cpu_access=True)
    self.cmdq_allocator = BumpAllocator(size=self.cmdq_page.size, base=int(self.cmdq_page.va_addr), wrap=True)
    self.cmdq = self.cmdq_page.cpu_view().view(fmt='I')

    self.num_gpcs, self.num_tpc_per_gpc, self.num_sm_per_tpc, self.max_warps_per_sm, self.sm_version = self._query_gpu_info('num_gpcs',
      'num_tpc_per_gpc', 'num_sm_per_tpc', 'max_warps_per_sm', 'sm_version')

    # FIXME: no idea how to convert this for blackwells
    self.arch: str = "sm_120" if self.sm_version==0xa04 else f"sm_{(self.sm_version>>8)&0xff}{(val>>4) if (val:=self.sm_version&0xff) > 0xf else val}"
    self.sass_version = ((self.sm_version & 0xf00) >> 4) | (self.sm_version & 0xf)

    compilers = CompilerSet(ctrl_var=NV_CC, cset=[(functools.partial(CUDARenderer, self.arch), None),
       (functools.partial(PTXRenderer, self.arch, device="NV"), NV_PTX),
       (functools.partial(NAKRenderer, self.arch, self.max_warps_per_sm), NV_NAK),
       (functools.partial(CUDARenderer, self.arch, use_nvcc=True), NV_NVCC)])
    super().__init__(device, NVAllocator(self), compilers, functools.partial(NVProgram, self), NVSignal, NVComputeQueue, NVCopyQueue)

    # On Tegra, force pushbuffer-based signal release to avoid QMD reuse race (fast MMIO doorbell outpaces GPU QMD reads).
    if self.is_tegra(): NVComputeQueue._tegra_signal = True

    self.pma_enabled = PMA.value > 0 and PROFILE >= 1 and not self.is_tegra()
    if self.pma_enabled: self._prof_init()

    self._setup_gpfifos()

  def _new_gpu_fifo(self, gpfifo_area, ctxshare, channel_group, offset=0, entries=0x400, compute=False, video=False) -> GPFifo:
    notifier = self.iface.alloc(48 << 20, uncached=True)
    params = nv_gpu.NV_CHANNELGPFIFO_ALLOCATION_PARAMETERS(gpFifoOffset=gpfifo_area.va_addr+offset, gpFifoEntries=entries, hContextShare=ctxshare,
      hObjectError=notifier.meta.hMemory, hObjectBuffer=self.virtmem if video else gpfifo_area.meta.hMemory,
      hUserdMemory=(ctypes.c_uint32*8)(gpfifo_area.meta.hMemory), userdOffset=(ctypes.c_uint64*8)(entries*8+offset), engineType=19 if video else 0)
    gpfifo = self.iface.rm_alloc(channel_group, self.iface.gpfifo_class, params)

    if compute:
      self.debug_compute_obj, self.debug_channel = self.iface.rm_alloc(gpfifo, self.iface.compute_class), gpfifo
      debugger_params = nv_gpu.NV83DE_ALLOC_PARAMETERS(hAppClient=self.iface.root, hClass3dObject=self.debug_compute_obj)
      self.debugger = self.iface.rm_alloc(self.nvdevice, nv_gpu.GT200_DEBUGGER, debugger_params)
    elif not video: self.iface.rm_alloc(gpfifo, self.iface.dma_class)
    else: self.iface.rm_alloc(gpfifo, self.iface.viddec_class)

    if channel_group == self.nvdevice:
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_BIND, nv_gpu.NVA06F_CTRL_BIND_PARAMS(engineType=params.engineType))
      self.iface.rm_control(gpfifo, nv_gpu.NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, nv_gpu.NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS(bEnable=1))

    ws_token_params = self.iface.rm_control(gpfifo, nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN,
      nv_gpu.NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS(workSubmitToken=-1))
    if ctxshare != 0: self.iface.setup_gpfifo_vm(gpfifo)

    return GPFifo(ring=gpfifo_area.cpu_view().view(offset, entries*8, fmt='Q'), entries_count=entries, token=ws_token_params.workSubmitToken,
                  gpput=gpfifo_area.cpu_view().view(offset + entries*8 + getattr(nv_gpu.AmpereAControlGPFifo, 'GPPut').offset, fmt='I'))

  def _query_gpu_info(self, *reqs):
    nvrs = [getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_'+r.upper(), getattr(nv_gpu,'NV2080_CTRL_GR_INFO_INDEX_LITTER_'+r.upper(), None)) for r in reqs]

    if self.is_nvd():
      x = self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO,
        nv_gpu.NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS())
      return [x.engineInfo[0].infoList[nvr].data for nvr in nvrs]

    infos = (nv_gpu.NV2080_CTRL_GR_INFO*len(nvrs))(*[nv_gpu.NV2080_CTRL_GR_INFO(index=nvr) for nvr in nvrs])
    self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_INFO,
      nv_gpu.NV2080_CTRL_GR_GET_INFO_PARAMS(grInfoListSize=len(infos), grInfoList=ctypes.addressof(infos)))
    return [x.data for x in infos]

  def _setup_gpfifos(self):
    self.slm_per_thread, self.shader_local_mem = 0, None

    # Set windows addresses to not collide with other allocated buffers.
    # Tegra has 40-bit VA space (max 0xFFFFFFFFFF), desktop has 43-bit+.
    if self.is_tegra():
      self.shared_mem_window, self.local_mem_window = 0xFE00000000, 0xFD00000000
    else:
      self.shared_mem_window, self.local_mem_window = 0x729400000000, 0x729300000000

    NVComputeQueue().setup(compute_class=self.iface.compute_class, local_mem_window=self.local_mem_window, shared_mem_window=self.shared_mem_window) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)

    NVCopyQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                 .setup(copy_class=self.iface.dma_class) \
                 .signal(self.timeline_signal, self.next_timeline()).submit(self)

    self.synchronize()

  def _ensure_has_local_memory(self, required):
    if self.slm_per_thread >= required: return

    self.slm_per_thread, old_slm_per_thread = round_up(required, 32), self.slm_per_thread
    bytes_per_tpc = round_up(round_up(self.slm_per_thread * 32, 0x200) * self.max_warps_per_sm * self.num_sm_per_tpc, 0x8000)
    self.shader_local_mem, ok = self._realloc(self.shader_local_mem, round_up(bytes_per_tpc*self.num_tpc_per_gpc*self.num_gpcs, 0x20000))

    # Realloc failed, restore the old value.
    if not ok: self.slm_per_thread = old_slm_per_thread

    cast(NVComputeQueue, NVComputeQueue().wait(self.timeline_signal, self.timeline_value - 1)) \
                                         .setup(local_mem=self.shader_local_mem.va_addr, local_mem_tpc_bytes=bytes_per_tpc) \
                                         .signal(self.timeline_signal, self.next_timeline()).submit(self)

  def _ensure_has_vid_hw(self, w, h):
    if self.iface.viddec_class is None: raise RuntimeError(f"{self.device} Video decoder class not available.")

    coloc_size = round_up((round_up(h, 64) * round_up(h, 64)) + (round_up(w, 64) * round_up(h, 64) // 16), 2 << 20)
    self.intra_top_off = round_up(h, 64) * (608 + 4864 + 152 + 2000)
    intra_unk_size = ((2 << 20) if self.iface.viddec_class >= nv_gpu.NVCFB0_VIDEO_DECODER else 0)
    self.intra_unk_off = (round_up(self.intra_top_off, 0x10000) + (64 << 10)) if intra_unk_size > 0 else None
    filter_size = round_up(round_up(self.intra_top_off, 0x10000) + (64 << 10) + intra_unk_size, 2 << 20)

    if not hasattr(self, 'vid_gpfifo'):
      self.vid_gpfifo = self._new_gpu_fifo(self.gpfifo_area, 0, self.nvdevice, offset=0x200000, entries=2048, compute=False, video=True)
      self.vid_coloc_buf, self.vid_filter_buf = self.allocator.alloc(coloc_size), self.allocator.alloc(filter_size)
      self.vid_stat_buf = self.allocator.alloc(0x1000)
      NVVideoQueue().wait(self.timeline_signal, self.timeline_value - 1) \
                    .setup(copy_class=self.iface.viddec_class) \
                    .signal(self.timeline_signal, self.next_timeline()).submit(self)
    else:
      if coloc_size > self.vid_coloc_buf.size: self.vid_coloc_buf, _ = self._realloc(self.vid_coloc_buf, coloc_size, force=True)
      if filter_size > self.vid_filter_buf.size: self.vid_filter_buf, _ = self._realloc(self.vid_filter_buf, filter_size, force=True)

  def invalidate_caches(self):
    if self.is_nvd(): self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_INTERNAL_BUS_FLUSH_WITH_SYSMEMBAR, None)
    else:
      self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_FB_FLUSH_GPU_CACHE, nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_PARAMS(
        flags=((nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_WRITE_BACK_YES << 2) | (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_INVALIDATE_YES << 3) |
              (nv_gpu.NV2080_CTRL_FB_FLUSH_GPU_CACHE_FLAGS_FLUSH_MODE_FULL_CACHE << 4))))

  def on_device_hang(self):
    # Prepare fault report.
    # TODO: Restore the GPU using NV83DE_CTRL_CMD_CLEAR_ALL_SM_ERROR_STATES if needed.

    if self.is_tegra():
      raise RuntimeError("GPU hang detected on Tegra device (no detailed fault info available via nvgpu)")

    report = []
    sm_errors = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_ALL_SM_ERROR_STATES,
      nv_gpu.NV83DE_CTRL_DEBUG_READ_ALL_SM_ERROR_STATES_PARAMS(hTargetChannel=self.debug_channel, numSMsToRead=100))

    if sm_errors.mmuFault.valid:
      mmu = self.iface.rm_control(self.debugger, nv_gpu.NV83DE_CTRL_CMD_DEBUG_READ_MMU_FAULT_INFO,
        nv_gpu.NV83DE_CTRL_DEBUG_READ_MMU_FAULT_INFO_PARAMS())
      for i in range(mmu.count):
        pfinfo = mmu.mmuFaultInfoList[i]
        report += [f"MMU fault: 0x{pfinfo.faultAddress:X} | {NV_PFAULT_FAULT_TYPE[pfinfo.faultType]} | {NV_PFAULT_ACCESS_TYPE[pfinfo.accessType]}"]
    else:
      for i, e in enumerate(sm_errors.smErrorStateArray):
        if e.hwwGlobalEsr or e.hwwWarpEsr: report += [f"SM {i} fault: esr={e.hwwGlobalEsr} warp_esr={e.hwwWarpEsr:#x} warp_pc={e.hwwWarpEsrPc64:#x}"]

    raise RuntimeError("\n".join(report))

  def _prof_init(self):
    self.profiler = self.iface.rm_alloc(self.subdevice, nv_gpu.MAXWELL_PROFILER_DEVICE,
      nv_gpu.NVB2CC_ALLOC_PARAMETERS(hClientTarget=self.iface.root, hContextTarget=self.channel_group))

    power_params = nv_gpu.struct_NVB0CC_CTRL_POWER_REQUEST_FEATURES_PARAMS(controlMask=(nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELCG_DISABLE << 0) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_BLCG_DISABLE << 2) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_ELPG_DISABLE << 6) | \
      (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_IDLE_SLOWDOWN_DISABLE << 8) | (nv_gpu.NVB0CC_CTRL_POWER_FEATURE_MASK_VAT_DISABLE << 10))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_POWER_REQUEST_FEATURES, power_params)

    self.pma_buf = self.iface.alloc(getenv("PMA_BUFFER_SIZE", 512) << 20, uncached=True, cpu_cached=True, cpu_access=True)
    self.pma_bytes = self.iface.alloc(0x1000, uncached=True, cpu_cached=True, cpu_access=self.is_nvd(), read_only=True)
    self.pma_rptr = 0

    pma_stream = nv_gpu.struct_NVB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS(hMemPmaBuffer=self.pma_buf.meta.hMemory,
      pmaBufferSize=self.pma_buf.size, hMemPmaBytesAvailable=self.pma_bytes.meta.hMemory, pmaBufferVA=self.pma_buf.va_addr)
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_ALLOC_PMA_STREAM, pma_stream, extra=(self.pma_buf, self.pma_bytes))

    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY, nv_gpu.struct_NVB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS(ctxsw=0))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER)
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_BIND_PM_RESOURCES)

    self._prof_setup_pc_sampling()

  def _prof_setup_pc_sampling(self):
    is_bw = self.iface.compute_class >= nv_gpu.BLACKWELL_COMPUTE_A
    PMASYS_BASE, PMAGPC_BASE, GR_GPC_BASE, GPC_BASE = (0x2b1000, 0x2b0000, 0x424000, 0x200000) if is_bw else (0x24a000, 0x244000, 0x419800, 0x180000)

    tpc_masks = [m for i in range(self.num_gpcs) if (m:=self.iface.rm_control(self.subdevice, nv_gpu.NV2080_CTRL_CMD_GR_GET_TPC_MASK,
      nv_gpu.NV2080_CTRL_GR_GET_TPC_MASK_PARAMS(gpcId=i)).tpcMask) > 0]
    tpc_cnt = [bin(mask).count('1') for mask in tpc_masks]

    # enables pma on gpc
    if not is_bw: self.reg_ops(*[(PMAGPC_BASE + gpc * 0x200, 0x100, 0x100) for gpc in range(len(tpc_masks))])

    # sets streaming bw for each gpc
    hs = nv_gpu.struct_NVB0CC_CTRL_HS_CREDITS_PARAMS(pmaChannelIdx=0, numEntries=len(tpc_masks))
    for i, mask in enumerate(tpc_masks):
      hs.creditInfo[i] = nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO(
        chipletType=nv_gpu.NVB0CC_CHIPLET_TYPE_GPC, chipletIndex=i, numCredits=bin(mask).count('1'))
    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_SET_HS_CREDITS, hs)

    if is_bw:
      # enables pma on gpcs
      self.reg_ops(*[op for i in range(3) for op in [(PMASYS_BASE + 0x128 + i*8, 480), (PMASYS_BASE + 0x12c + i*8, 0x80000000)]])
      self.reg_ops((PMAGPC_BASE + 0xa24, 0x04000001), (PMAGPC_BASE + 0xa10, 0x80000002))
      self.reg_ops(*[(GPC_BASE + gpc * 0x4000 + 0x200 + tpc * 0x200 + reg, 0)
                     for gpc in range(len(tpc_masks)) for tpc in range(tpc_cnt[gpc]) for reg in [0x100, 0x108, 0x110, 0x120]])

      def SM_REG(gpc, tpc, sm, reg): return GPC_BASE + gpc * 0x4000 + 0x800 + (tpc * self.num_sm_per_tpc + sm) * 0x200 + reg
    else:
      self.reg_ops(*[(PMASYS_BASE + 0x65c + off * 4, 0xffffffff) for off in range(self.num_gpcs * 2)])
      self.reg_ops((PMASYS_BASE + 0x620, 0x2000007))

      def SM_REG(gpc, tpc, sm, reg): return GPC_BASE + gpc * 0x4000 + (self.num_tpc_per_gpc - tpc_cnt[gpc] + tpc) * 0x200 + [0x400, 0x1000][sm] + reg

    # enable pc sampling for the context
    self.reg_ops((GR_GPC_BASE + 0x304, 0x80808a))

    # sm config and enable
    self.reg_ops(*[op for gpc in range(len(tpc_masks)) for tpc in range(tpc_cnt[gpc]) for sm in range(self.num_sm_per_tpc) for op in [
      (SM_REG(gpc, tpc, sm, 0x128), (gpc << 5) | (tpc << 1) | sm), # enumeration. NOTE: different from cuda
      (SM_REG(gpc, tpc, sm, 0x40), 0x19181716), (SM_REG(gpc, tpc, sm, 0x48), 0x1d1c1b1a), (SM_REG(gpc, tpc, sm, 0x50), 0x1e201f), # unk, counters?
      (SM_REG(gpc, tpc, sm, 0xec), 0x1), (SM_REG(gpc, tpc, sm, 0x6c), 0x2), (SM_REG(gpc, tpc, sm, 0x9c), 0x5),
      (SM_REG(gpc, tpc, sm, 0x108), 0xa0 if is_bw else 0x20), *([(SM_REG(gpc, tpc, sm, 0x120), 0x100000)] if is_bw else [])]])
    self.reg_ops((GR_GPC_BASE + 0x3dc, 0x1), reg_type=1)

  def reg_ops(self, *ops, reg_type=0, op=nv_gpu.NV2080_CTRL_GPU_REG_OP_WRITE_32):
    for i in range(0, len(ops), 124):
      params = nv_gpu.struct_NVB0CC_CTRL_EXEC_REG_OPS_PARAMS(regOpCount=len(chunk:=ops[i:i+124]))
      for j, (off, val, *rest) in enumerate(chunk):
        params.regOps[j] = nv_gpu.struct_NV2080_CTRL_GPU_REG_OP(regOp=op, regType=reg_type,
          regOffset=off, regValueLo=val, regAndNMaskLo=rest[0] if rest else 0xffffffff)
      with contextlib.suppress(RuntimeError): self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_EXEC_REG_OPS, params)

  def _prof_readback(self) -> bytes|None:
    params = self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT,
      nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bUpdateAvailableBytes=1, bWait=1))

    if params.bOverflowStatus: raise RuntimeError("PMA profiler: buffer overflow detected")
    if params.bytesAvailable == 0: return None

    start, end = self.pma_rptr, self.pma_rptr + params.bytesAvailable
    pma_data = self.pma_buf.cpu_view()[start:min(end, self.pma_buf.size)] + self.pma_buf.cpu_view()[:max(0, end - self.pma_buf.size)]
    self.pma_rptr = end % self.pma_buf.size

    self.iface.rm_control(self.profiler, nv_gpu.NVB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT,
      nv_gpu.struct_NVB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS(bytesConsumed=params.bytesAvailable))
    return pma_data

  def device_props(self): return {'arch': self.arch, 'sm_version': self.sm_version}
