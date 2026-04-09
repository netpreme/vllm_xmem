## Simulated coding agent runs

# ── MTier (GPU 0, port 8000) 
─────────────────────────────────────  
```                          
mtier_service reset
rm -rf /tmp/dynamo_store_kv_8000                                 
clear && CUDA_VISIBLE_DEVICES=0 DYNAMO_PORT=8000 bash /home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/launch_dynamo.sh --mtier > /tmp/launch_8000.log 2>&1         
tail -f /tmp/launch_8000.log                                     
# (wait until 'generate' appears in health, then run test)     
clear && python3 '/home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/stress_kv_offload.py' --url http://localhost:8000 --log /tmp/dynamo_worker_8000.log --fill 3 --turns 4 --evict 1 --recall 3                                   
```


# ── Kill MTier before starting CPU                              
──────────────────────────────
```
ps aux | grep -E "(dynamo|VLLM)" | grep -v grep | awk '{print $2}' | while read pid; do kill -9 $pid 2>/dev/null; done; sleep 3
```                          
                            
# ── CPU DRAM (GPU 1, port 8001)                                 
──────────────────────────────────
rm -rf /tmp/dynamo_store_kv_8001                                 
clear && CUDA_VISIBLE_DEVICES=1 DYNAMO_PORT=8001 bash /home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/launch_dynamo.sh > /tmp/launch_8001.log 2>&1
tail -f /tmp/launch_8001.log                                     
# (wait until 'generate' appears in health, then run test)     
clear && python3 '/home/ubuntu/vllm_xmem/netpreme/prefix_caching/anthropic/_dynamo/stress_kv_offload.py' --url http://localhost:8001 --log /tmp/dynamo_worker_8001.log --fill 3 --turns 4 --evict 1 --recall 3 

