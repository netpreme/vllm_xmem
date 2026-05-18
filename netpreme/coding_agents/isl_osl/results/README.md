# ISL/OSL benchmark results

Per-turn input/output sequence-length distributions for code-agent workloads on SWE-bench verified.

## Reproduction

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash ../server.sh > /tmp/vllm.log 2>&1 &            # vLLM only
./run.sh                                            # results saved in ./runs/<stamp>
./analyze.sh runs/<stamp>                           # figure generation
```
