"""
For each GPU: classify as CPU-attached (PCIe only) or XMem-attached (NVLink),
and identify which XMem bank each GPU is home to.

Measurements:
  1. CPU bandwidth  — pinned H2D/D2H via PCIe (~55 GB/s on H100 SXM5)
  2. XMem bandwidth — cudaMemcpyDeviceToDevice to each bank (G→X and X→G)

Home GPU detection (authoritative):
  mtier_sdk.malloc(bank_id) returns a CUDA device pointer.
  cudaPointerGetAttributes(ptr).device tells which GPU physically owns it —
  this is the home GPU for that bank, regardless of which context did the malloc.

Expected numbers (H100 SXM5, NV18 topology):
  CPU  H2D/D2H : ~55 GB/s       (PCIe Gen4 x16)
  XMem home    : ~1,400+ GB/s   (direct HBM3)
  XMem remote  : ~350–450 GB/s  (NVSwitch hop)
"""
import torch
import time
import os

# ── XMem / MTier availability ─────────────────────────────────────────────────
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

# ── config ────────────────────────────────────────────────────────────────────
SIZE = 256 * 1024 ** 2   # 256 MB per transfer
REPS = 10


def _bw(seconds: float) -> float:
    return REPS * SIZE / seconds / 1e9


def xmem_home_gpu(bank_id: int) -> int:
    """
    Return the GPU ordinal that physically owns this XMem bank.

    We allocate a tiny buffer in the bank (any GPU's context works — we use
    GPU 0), then ask CUDA's UVA system which device the pointer belongs to via
    cudaPointerGetAttributes().  The returned attrs.device is the home GPU.

    This is the authoritative answer: CUDA tracks physical device ownership
    through UVA regardless of which context performed the allocation.
    """
    if not (XMEM_AVAILABLE and CUDART_AVAILABLE):
        return -1
    torch.cuda.set_device(0)
    _ = torch.empty(1, device="cuda:0")  # ensure context exists
    try:
        ptr = mtier_sdk.malloc(bank_id, 4096)
        err, attrs = cudart.cudaPointerGetAttributes(ptr)
        mtier_sdk.free(ptr)
        if err == cudart.cudaError_t.cudaSuccess:
            return attrs.device
    except Exception as e:
        print(f"  [xmem] cudaPointerGetAttributes failed for bank{bank_id}: {e}")
    return -1


def bench_cpu(gpu_id: int):
    """H2D and D2H via pinned CPU memory (PCIe path)."""
    dev = f"cuda:{gpu_id}"
    n = SIZE // 4
    dst = torch.empty(n, dtype=torch.float32, device=dev)
    src = torch.randn(n, dtype=torch.float32).pin_memory()

    dst.copy_(src); torch.cuda.synchronize(dev)  # warm-up

    t0 = time.perf_counter()
    for _ in range(REPS):
        dst.copy_(src)
    torch.cuda.synchronize(dev)
    h2d = _bw(time.perf_counter() - t0)

    dst2 = torch.empty(n, dtype=torch.float32).pin_memory()
    t0 = time.perf_counter()
    for _ in range(REPS):
        dst2.copy_(dst)
    torch.cuda.synchronize(dev)
    d2h = _bw(time.perf_counter() - t0)

    return h2d, d2h


def bench_xmem_bank(gpu_id: int, bank_id: int):
    """
    Measure GPU ↔ XMem D2D bandwidth for a specific bank.
    Returns (g2x_gb, x2g_gb) or None on failure.
    """
    if not (XMEM_AVAILABLE and CUDART_AVAILABLE):
        return None

    dev = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    _ = torch.empty(1, device=dev)  # ensure CUDA context exists

    try:
        remote_ptr = mtier_sdk.malloc(bank_id, SIZE)
    except Exception:
        return None

    src = torch.empty(SIZE // 4, dtype=torch.float32, device=dev)
    dst = torch.empty(SIZE // 4, dtype=torch.float32, device=dev)

    try:
        # warm-up
        cudart.cudaMemcpy(remote_ptr, src.data_ptr(), SIZE,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)

        # GPU → XMem
        t0 = time.perf_counter()
        for _ in range(REPS):
            cudart.cudaMemcpy(remote_ptr, src.data_ptr(), SIZE,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)
        g2x = _bw(time.perf_counter() - t0)

        # XMem → GPU
        t0 = time.perf_counter()
        for _ in range(REPS):
            cudart.cudaMemcpy(dst.data_ptr(), remote_ptr, SIZE,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
        torch.cuda.synchronize(dev)
        x2g = _bw(time.perf_counter() - t0)

        return g2x, x2g

    except Exception as e:
        print(f"  [xmem] D2D error GPU{gpu_id}→bank{bank_id}: {e}")
        return None
    finally:
        try:
            mtier_sdk.free(remote_ptr)
        except Exception:
            pass


def _gb(n: int) -> str:
    return f"{n / 1024**3:.1f} GB"


# ── main ──────────────────────────────────────────────────────────────────────
num_gpus = torch.cuda.device_count()

# System CPU RAM
cpu_total_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3

# XMem bank sizes
xmem_bank_sizes: dict[int, tuple[int, int]] = {}  # bank_id → (total, available)
if XMEM_AVAILABLE:
    for bank_id in range(NUM_BANKS):
        try:
            s = mtier_sdk.get_bank_status(bank_id)
            xmem_bank_sizes[bank_id] = (s.total_memory, s.available_memory)
        except Exception:
            xmem_bank_sizes[bank_id] = (0, 0)

print(f"GPUs found   : {num_gpus}")
print(f"CPU RAM      : {cpu_total_gb:.1f} GB  (shared across all GPUs via PCIe)")
if XMEM_AVAILABLE:
    for bank_id, (total, avail) in xmem_bank_sizes.items():
        print(f"XMem bank{bank_id}   : {_gb(total)} total  {_gb(avail)} free")
else:
    print(f"XMem service : down / unavailable")

# ── authoritative: which GPU owns each bank (via cudaPointerGetAttributes) ────
home_gpu_per_bank: dict[int, int] = {}
if XMEM_AVAILABLE and CUDART_AVAILABLE:
    print()
    print("XMem bank → home GPU  (via cudaPointerGetAttributes):")
    for bank_id in range(NUM_BANKS):
        hg = xmem_home_gpu(bank_id)
        home_gpu_per_bank[bank_id] = hg
        name = torch.cuda.get_device_name(hg) if hg >= 0 else "unknown"
        print(f"  bank{bank_id}  →  GPU{hg} [{name}]")
print()

# ── per-GPU bandwidth measurements ───────────────────────────────────────────
results: dict[int, dict] = {}
for gpu_id in range(num_gpus):
    h2d, d2h = bench_cpu(gpu_id)
    results[gpu_id] = {"h2d": h2d, "d2h": d2h, "banks": {}}
    for bank_id in range(NUM_BANKS):
        results[gpu_id]["banks"][bank_id] = bench_xmem_bank(gpu_id, bank_id)

# ── build table rows ─────────────────────────────────────────────────────────
rows = []
for gpu_id in range(num_gpus):
    h2d = results[gpu_id]["h2d"]
    d2h = results[gpu_id]["d2h"]
    gpu_mem_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3

    bank_cells: dict[int, tuple[str, str, str]] = {}  # bank_id → (size, g2x, x2g)
    home_banks = []
    for bank_id in range(NUM_BANKS):
        bw = results[gpu_id]["banks"].get(bank_id)
        is_home = (home_gpu_per_bank.get(bank_id) == gpu_id)
        if is_home:
            home_banks.append(bank_id)
        total, _ = xmem_bank_sizes.get(bank_id, (0, 0))
        tag = "HOME" if is_home else "NVSw"
        if bw is None:
            bank_cells[bank_id] = (_gb(total), "N/A", "N/A")
        else:
            bank_cells[bank_id] = (_gb(total), f"{bw[0]:.0f}", f"{bw[1]:.0f}")

    any_xmem = any(results[gpu_id]["banks"].get(b) is not None for b in range(NUM_BANKS))
    if NUM_BANKS == 0 or not any_xmem:
        connection = "CPU only"
    elif home_banks:
        connection = f"XMem HOME  bank{home_banks[0]}"
    else:
        connection = "XMem NVSwitch"

    rows.append({
        "gpu":        f"GPU{gpu_id}",
        "hbm":        f"{gpu_mem_gb:.1f} GB",
        "cpu_ram":    f"{cpu_total_gb:.1f} GB",
        "h2d":        f"{h2d:.1f}",
        "d2h":        f"{d2h:.1f}",
        "banks":      bank_cells,
        "home_banks": home_banks,
        "connection": connection,
    })

# ── render table ──────────────────────────────────────────────────────────────
# Fixed columns
cols = [
    ("GPU",        "gpu",     6),
    ("HBM",        "hbm",     8),
    ("CPU RAM",    "cpu_ram", 8),
    ("H2D GB/s",   "h2d",     9),
    ("D2H GB/s",   "d2h",     9),
]
# Dynamic XMem bank columns (G→X and X→G per bank)
for bank_id in range(NUM_BANKS):
    total, _ = xmem_bank_sizes.get(bank_id, (0, 0))
    label = f"XMem b{bank_id} {_gb(total)}"
    cols.append((f"{label} G→X",  f"b{bank_id}_g2x", 16))
    cols.append((f"{label} X→G",  f"b{bank_id}_x2g", 16))
cols.append(("Connection", "connection", 22))

# Flatten bank cells into row dicts
for row in rows:
    for bank_id, (size, g2x, x2g) in row["banks"].items():
        row[f"b{bank_id}_g2x"] = f"{g2x} GB/s"
        row[f"b{bank_id}_x2g"] = f"{x2g} GB/s"

def _sep(char="─"):
    return "┼".join(char * (w + 2) for _, _, w in cols)

header = "│".join(f" {label:^{w}} " for label, _, w in cols)
divider = _sep()
print("┌" + _sep("─").replace("┼", "┬") + "┐")
print("│" + header + "│")
print("├" + divider.replace("┼", "┼") + "┤")
for row in rows:
    line = "│".join(f" {row.get(key, ''):^{w}} " for _, key, w in cols)
    print("│" + line + "│")
print("└" + _sep("─").replace("┼", "┴") + "┘")
