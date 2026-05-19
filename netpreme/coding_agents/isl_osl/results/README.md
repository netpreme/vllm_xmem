# ISL/OSL benchmark results

Per-turn input/output sequence-length distributions for code-agent workloads on SWE-bench verified.

## Files

- `analysis_dist_grid.png` — OSL / ISL / ISL_uncached histograms organized by SWE-bench difficulty bucket. Total of 500 problems
- `analysis_dist_agg.png` —  Aggregated OSL / ISL / ISL_uncached histograms of the above
- `analysis_dist_agg.npz` — Raw per-turn values + binned counts + shared bin edges, loadable with `numpy.load`.
- `analysis_cache.png` — vLLM prefix-cache hit rate per turn, bucketed by difficulty.

## Reproduction

```bash
cd /root/vllm_xmem/netpreme/coding_agents/isl_osl
bash ../server.sh > /tmp/vllm.log 2>&1 &            # vLLM only
./run.sh                                            # results saved in ./runs/<stamp>
./analyze.sh runs/<stamp>                           # figure generation
```
