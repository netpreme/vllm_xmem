"""
Bandwidth benchmark: CPU (PCIe), XMem bank0, XMem bank1.

Measured on this machine (2x H100 80GB HBM3, NV18 topology):
    CPU  H2D/D2H  :  ~55 GB/s   (PCIe Gen4 x16)
    XMem bank0    : ~396 GB/s   (NVLink fabric, 18 lanes x 26.562 GB/s = ~478 GB/s ceiling)
    XMem bank1    : ~362 GB/s   (NVLink fabric)

Usage:
    python3 benchmark_effective_bandwidth.py                    # all, GPU 0
    python3 benchmark_effective_bandwidth.py --mode cpu         # CPU DRAM only
    python3 benchmark_effective_bandwidth.py --mode xmem        # XMem banks only
    python3 benchmark_effective_bandwidth.py --gpu 1            # run on GPU 1
    python3 benchmark_effective_bandwidth.py --size 512 --reps 20
"""
import argparse
import os
import time

import torch

# ── XMem availability ─────────────────────────────────────────────────────────
XMEM_AVAILABLE = False
NUM_BANKS = 0

try:
    from xmem import mtier_sdk
    LIB = os.getenv("MTIER_LIB_PATH", "/usr/local/lib/libmtier.so")
    mtier_sdk.init(LIB)
    _st = mtier_sdk.get_status()
    XMEM_AVAILABLE = (_st.status == 0)
    NUM_BANKS = _st.num_banks if XMEM_AVAILABLE else 0
except Exception as _e:
    print(f"[xmem] not available: {_e}\n")

try:
    from cuda.bindings import runtime as cudart
    CUDART_AVAILABLE = True
except ImportError:
    CUDART_AVAILABLE = False
    print("[cuda.bindings] not available — XMem benchmarks require cuda-python\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _bw(total_bytes: int, seconds: float) -> float:
    return total_bytes / seconds / 1e9


def _gb(n: int) -> str:
    return f"{n / 1024**3:.1f} GB"


def _row(label: str, g2m: float | None, m2g: float | None):
    g = f"{g2m:6.1f} GB/s" if g2m is not None else "     N/A  "
    m = f"{m2g:6.1f} GB/s" if m2g is not None else "     N/A  "
    print(f"  {label:<36}  GPU→mem: {g}    mem→GPU: {m}")


# ── benchmarks ────────────────────────────────────────────────────────────────

def bench_cpu(gpu_id: int, size: int, reps: int) -> tuple[float, float]:
    """H2D and D2H via pinned CPU memory (PCIe path)."""
    dev = f"cuda:{gpu_id}"
    n = size // 4

    gpu_buf = torch.empty(n, dtype=torch.float32, device=dev)
    cpu_buf = torch.randn(n, dtype=torch.float32).pin_memory()
    cpu_out = torch.empty(n, dtype=torch.float32).pin_memory()

    # warm-up
    gpu_buf.copy_(cpu_buf)
    torch.cuda.synchronize(dev)

    t0 = time.perf_counter()
    for _ in range(reps):
        gpu_buf.copy_(cpu_buf)
    torch.cuda.synchronize(dev)
    h2d = _bw(reps * size, time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(reps):
        cpu_out.copy_(gpu_buf)
    torch.cuda.synchronize(dev)
    d2h = _bw(reps * size, time.perf_counter() - t0)

    return h2d, d2h


def bench_xmem_bank(gpu_id: int, bank_id: int, size: int, reps: int) -> tuple[float, float] | None:
    """GPU ↔ XMem via cudaMemcpyDeviceToDevice over NVLink fabric."""
    if not (XMEM_AVAILABLE and CUDART_AVAILABLE):
        return None

    dev = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    _ = torch.empty(1, device=dev)

    try:
        remote_ptr = mtier_sdk.malloc(bank_id, size)
    except Exception as e:
        print(f"  [xmem] malloc failed bank{bank_id}: {e}")
        return None

    gpu_buf = torch.empty(size // 4, dtype=torch.float32, device=dev)

    try:
        # warm-up
        cudart.cudaMemcpy(remote_ptr, gpu_buf.data_ptr(), size,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)

        # GPU → XMem
        t0 = time.perf_counter()
        for _ in range(reps):
            cudart.cudaMemcpy(remote_ptr, gpu_buf.data_ptr(), size,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)
        g2x = _bw(reps * size, time.perf_counter() - t0)

        # XMem → GPU
        t0 = time.perf_counter()
        for _ in range(reps):
            cudart.cudaMemcpy(gpu_buf.data_ptr(), remote_ptr, size,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)
        x2g = _bw(reps * size, time.perf_counter() - t0)

        return g2x, x2g

    except Exception as e:
        print(f"  [xmem] D2D error GPU{gpu_id} ↔ bank{bank_id}: {e}")
        return None
    finally:
        try:
            mtier_sdk.free(remote_ptr)
        except Exception:
            pass


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "xmem", "all"],
        default="all",
        help="Which path to benchmark (default: all)",
    )
    parser.add_argument("--gpu",  type=int, default=0,   help="GPU ordinal (default: 0)")
    parser.add_argument("--size", type=int, default=256, metavar="MB", help="Transfer size MB (default: 256)")
    parser.add_argument("--reps", type=int, default=10,  help="Repetitions (default: 10)")
    args = parser.parse_args()

    size   = args.size * 1024 * 1024
    gpu_id = args.gpu

    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        print(f"ERROR: GPU {gpu_id} not found ({num_gpus} available)")
        return

    gpu_name = torch.cuda.get_device_name(gpu_id)
    gpu_mem  = torch.cuda.get_device_properties(gpu_id).total_memory

    print(f"\nBandwidth Benchmark")
    print(f"  GPU {gpu_id}         : {gpu_name}  ({_gb(gpu_mem)} HBM)")
    print(f"  Transfer size : {args.size} MB")
    print(f"  Repetitions   : {args.reps}")

    if XMEM_AVAILABLE:
        print(f"\nXMem: {NUM_BANKS} bank(s)")
        for bank_id in range(NUM_BANKS):
            try:
                s = mtier_sdk.get_bank_status(bank_id)
                print(f"  bank{bank_id}: {_gb(s.total_memory)} total  {_gb(s.available_memory)} free")
            except Exception:
                print(f"  bank{bank_id}: (status unavailable)")
    else:
        print(f"\nXMem: not available")

    nvlink_ceiling = 18 * 26.562  # 18 lanes × 26.562 GB/s
    print(f"\nNVLink fabric ceiling : 18 lanes × 26.562 GB/s = {nvlink_ceiling:.0f} GB/s")
    print(f"PCIe Gen4 x16 ceiling : ~64 GB/s")
    print(f"\n{'─'*68}")

    if args.mode in ("cpu", "all"):
        print(f"\n[CPU DRAM — PCIe Gen4 x16]")
        h2d, d2h = bench_cpu(gpu_id, size, args.reps)
        _row(f"GPU{gpu_id} ↔ CPU DRAM", h2d, d2h)

    if args.mode in ("xmem", "all"):
        print(f"\n[XMem — NVLink fabric  (ceiling ~{nvlink_ceiling:.0f} GB/s)]")
        if not XMEM_AVAILABLE:
            print("  XMem not available")
        else:
            for bank_id in range(NUM_BANKS):
                result = bench_xmem_bank(gpu_id, bank_id, size, args.reps)
                if result:
                    _row(f"GPU{gpu_id} ↔ XMem bank{bank_id}", *result)
                else:
                    print(f"  GPU{gpu_id} ↔ bank{bank_id}: failed")

    print(f"\n{'─'*68}\n")


if __name__ == "__main__":
    main()
