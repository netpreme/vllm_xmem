## Simulated coding agent runs

# ── Saturation mode (maximum bandwidth, measured decode baseline, deterministic)
────────────────────────────────────────────────────────────────────────────────
# --turns 14:       fills ~93% of GPU KV cache (~2231 blocks = 9.35 GB) → maximizes transfer
# --evict 4:        4 sessions × 4 evict-turns = ~2484 blocks > 2387 GPU capacity
#                   → all fill blocks pushed to CPU with minimal evict overhead
# --evict-turns 4:  evict sessions only need enough blocks to overflow GPU; 4 turns is enough
# --pure-bw:        1-token recall, temperature=0, seed=42 → greedy/deterministic
#                   automatically runs GPU-hit baseline AFTER recall to measure true
#                   decode overhead.  True transfer time = recall_TTFT − baseline_TTFT.
# --fill 1 --recall 1: single session, clean single measurement


# ── MTIER (GPU 0, port 8000)                                 
──────────────────────────────────
```
nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=0 | xargs -r kill -9 && mtier_service reset && sleep 3

clear && CUDA_VISIBLE_DEVICES=0 DYNAMO_PORT=8000 bash /home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/launch_dynamo.sh --mtier > /tmp/launch_8000.log 2>&1 & tail -f /tmp/launch_8000.log
clear && python3 '/home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/stress_kv_offload.py' --url http://localhost:8000 --log /tmp/dynamo_worker_8000.log --evict-turns 4 --fill 12 --turns 4 --evict 15  --recall 6 --pure-bw
```    
                            
# ── CPU DRAM (GPU 1, port 8001)                                 
──────────────────────────────────
mtier_service reset && rm -rf /tmp/dynamo_store_kv_8001 
nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=1 | xargs -r kill -9
clear && CUDA_VISIBLE_DEVICES=1 DYNAMO_PORT=8001 bash /home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/launch_dynamo.sh > /tmp/launch_8001.log 2>&1 & tail -f /tmp/launch_8001.log                                     
clear && python3 '/home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/stress_kv_offload.py' --url http://localhost:8001 --log /tmp/dynamo_worker_8001.log --evict-turns 4 --fill 12 --turns 4 --evict 15  --recall 6 --pure-bw



