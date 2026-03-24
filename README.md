<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=45%>
  </picture>
  <span style="font-size: 3em; font-weight: bold; vertical-align: middle; margin: 10px; position: relative; bottom: 25px; left:-17px;">+</span>
  <img alt="Netpreme" src="assets/netpreme_logo.png" width=40%>
</p>

<h3 align="center">
vLLM meets Netpreme's scale-up GPU memory expansion </h3>

🔥 We have built a prototype platform to enable developers and researchers to explore the use cases of scale-up GPU memory expansion. Please [contact](https://netpreme.com/developer) us to get access to it. 

---

## About
This repository is a fork of vLLM (v0.18.0) that integrates Netpreme’s X-Mem—a GPU memory expansion system—as a dedicated tier for KV cache storage. By replacing traditional CPU DRAM with X-Mem in the KV offloading module, we leverage ~10x higher bandwidth to bypass standard memory bottlenecks. This allows the system to reduce Time to First Token (TTFT) and achieve higher throughput and concurrency for KV-intensive workloads, such as multi-turn coding agents.

## Getting Started
Install vLLM+MTier from source

```bash
uv venv --python 3.12
git clone <url-to-repo>
cd vllm_mtier
uv pip install -e .
```

> [!NOTE]
> vLLM+MTier will only work on Netpreme's X-Mem VMs. Please [contact](https://netpreme.com/developer) us to get access to it.

## Usage

To replace CPU DRAM with X-Mem
* `vllm serve` CLI API: add `--kv-offloading-mtier` flag
* `LLM()` Python API: set the `kv_offload_mtier` argument to `True`.
  e.g.,
  ```python
  ktc = KVTransferConfig(
    kv_connector="OffloadingConnector",
    kv_role="kv_both",
    # The below config will be applied to X-Mem 
    #   instead of CPU DRAM if X-Mem is enabled
    kv_connector_extra_config={
        "block_size": CPU_BLOCK_SIZE,
        "cpu_bytes_to_use": cpu_bytes_to_use,
        "num_cpu_blocks": num_cpu_blocks, 
    }
  )
  llm = LLM(
      model=MODEL,
      block_size=GPU_BLOCK_SIZE,
      ...
      enable_prefix_caching=True,
      kv_transfer_config=ktc,
      kv_offload_mtier=True,
  )
  ```