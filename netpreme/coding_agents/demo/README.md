# Demo

The demo is used to show ttft improvement using Mtier, comparing with CPU. Claude code is connected to vllm, using the offload tier. Two servers are used for comparison: **hybrid-cpu** (DRAM offload) and **hybrid-mtier** (MTier chip offload), using the [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) problems through a KV cache stress test.

| Phase | Turn | Prompt size | Purpose |
|-------|------|-------------|---------|
| Fill-HBM | 1 | ~64K tokens | Loads KV cache into GPU HBM |
| Spill-to-offload | 2 | +64–127K tokens | Overflows HBM → spills KV to external memory |
| Recall | 3+ | Short (~10 tokens) | Short query that forces a full KV reload from external memory |

To see the effects of the offloading tier, we stress test the system to force the usage of KV offloading, broken down into three phases. Phase 1 is used to used to saturate the HBM memory. Phase 2 is used to write to the offload tier, and phase 3 is used to fetch from the offloading tier. The measurements from phase 3 are used to measure ttft.


## Setup

```bash
pip install datasets httpx
```

## How to run

Start the vLLM servers for both Mtier and CPU offload
```
bash netpreme/coding_agents/demo/start_servers.sh
```

Once the two instances of vLLM are ready, run on another terminal
```
python3 netpreme/coding_agents/demo/demo_ttft_compare.py
```

Options:
```
python3 demo/demo_ttft_compare.py -c 4    # 4 concurrent sessions (default 8)
python3 demo/demo_ttft_compare.py -r 5    # 5 recall turns (default 3)
```

The expected speed up is about 1.1x


## Monitoring

Once the servers are running, go to `http://localhost:3000/`. Navigate to Dashboards -> MTier vs CPU — Live Demo to check the live metrics. 

---

## Theoretical analysis

To a first approximation, ttft is measured by 
```
ttft = t_(queue) + t_(transfer) + t_(compute)
```

Suppose in the simple case, where the vllm's queue is not saturated, then the theoretical speed up comparing Mtier and CPU will be 

```
t_(transfer, CPU) + t_(compute) / t_(transfer, mtier) + t_(compute)

```

The max speed up using Mtier will be if its transfer time is 0, or inf bandwidth:
```
(t_(transfer, CPU) + t_(compute))/ t_(compute)
= t_(transfer, CPU) / t_(compute) + 1

```

the faster the t_compute, the better the ratio. 

CPUs bandwidth is connected by PCIe, roughly 100GB/s bidirectional, so 50GB/s unidirectional
Using `qwen3-coder-30b-a3b-instruct-fp8` as the model, from the Chinchilla paper, the flops can be approximated as 2 × Active_Params × Tokens. If we generate just 1 token, we have 2 * 3.3b * 1 = 6.6e9 FLOPs. 
Using FP8 tensor cores, on H100, we have ~4000 TFLOPS, so 1.65 us per token to compute in an ideal setting. 

The KV cache size per token is 2 * layers * heads * head_dim * bytes_fp = 2 * 48 * 4 * 128 * 1 = 49152 bytes per token.

```
t_(transfer, CPU) / t_(compute) + 1
= (49152 / 50e9) / 1.65e-6 + 1
≈ 1 + 0.98 / 1.65
= 1.59
```

The theoretical maximum speedup of MTier over CPU, under these assumptions, is about 1.59×.

---

## Emperical Analysis

From the benchmarking results, empirical transfer time using Mtier is about 3x faster. Using the measured 30GB/s for CPU bandwidth, 49152 bytes per token, we get `t_(tranfer, CPU) = 49152 / 30GB = 1.64us` per token to transfer. 

```
speedup = (t_(transfer, CPU) + t_(compute)) / (t_(transfer, Mtier) + t_(compute))
= (t_(transfer, CPU) + t_(compute)) / (t_(transfer, CPU) / 3 + t_(compute))
```

Using:
```
t_(compute) = 1.65us
CPU bandwidth = 30 GB/s
Mtier bandwidth = 3 x 30 = 90 GB/s
KV size per token = 49152 bytes

t_(transfer, CPU) = 49152 / 30e9 = 1.6384us
t_(transfer, Mtier) = 49152 / 90e9 = 0.54us

---
speedup = (1.6384 + 1.65) / (0.54 + 1.65)
= 1.50x
```

So with 3x faster bandwidth, the estimated speedup is about 1.50x for a single token for compute in prefill and a single token to fetch from the offloading tier.

---



