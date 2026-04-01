Running vllm server and client making requests

To make a chat

On separate processes, 
`./launch_vllm.sh`
`./infer_chat.sh`

---

To run conversation

set env to 

```
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
HOST=0.0.0.0
PORT=8000
TENSOR_PARALLEL_SIZE=1
DTYPE=auto
# MAX_MODEL_LEN=65536
GPU_MEMORY_UTILIZATION=0.90
CUDA_VISIBLE_DEVICES=7

```
`./launch_vllm.sh`
`./infer_conversation.sh`

vllm_xmem) cloud-user@gpu-h100-29:~/george/vllm_xmem$ bash '/home/cloud-user/george/vllm_xmem/prefix_caching/anthropic/infer_conversation.sh'
Multi-turn conversation (prefix caching enabled on server).
Type your message and press Enter. Ctrl-D or 'exit' to quit.
---

[you]: hello

[assistant]: Hello! How can I assist you today?

  ┌─ Request Tokens ─────────────────────────────────────────────
  │ Input tokens (this request):  30
  │ Output tokens (this request): 10
  ├─ Prefix Cache (cumulative, all requests) ────────────────────
  │ Queried tokens:               30.0
  │ Cache hit tokens:             0.0
  │ Cache hit rate:               0.0%
  │ KV cache reads  (from cache): 0.0 tokens
  │ KV cache writes (to cache):   30 tokens
  ├─ KV Cache Pool ──────────────────────────────────────────────
  │ Total GPU blocks:             1982 (block_size=16 tokens)
  │ Max token capacity:           31712
  └──────────────────────────────────────────────────────────────

[you]: hello

[assistant]: Hello again! Is there something specific you'd like to talk about or ask?

  ┌─ Request Tokens ─────────────────────────────────────────────
  │ Input tokens (this request):  50
  │ Output tokens (this request): 17
  ├─ Prefix Cache (cumulative, all requests) ────────────────────
  │ Queried tokens:               80.0
  │ Cache hit tokens:             32.0
  │ Cache hit rate:               40.0%
  │ KV cache reads  (from cache): 32.0 tokens
  │ KV cache writes (to cache):   48 tokens
  ├─ KV Cache Pool ──────────────────────────────────────────────
  │ Total GPU blocks:             1982 (block_size=16 tokens)
  │ Max token capacity:           31712
  └──────────────────────────────────────────────────────────────

[you]: hello

[assistant]: Hello! It seems you're saying hello quite a bit today. How can I assist you? Do you have any questions or topics you'd like to discuss?

  ┌─ Request Tokens ─────────────────────────────────────────────
  │ Input tokens (this request):  77
  │ Output tokens (this request): 33
  ├─ Prefix Cache (cumulative, all requests) ────────────────────
  │ Queried tokens:               157.0
  │ Cache hit tokens:             96.0
  │ Cache hit rate:               61.1%
  │ KV cache reads  (from cache): 96.0 tokens
  │ KV cache writes (to cache):   61 tokens
  ├─ KV Cache Pool ──────────────────────────────────────────────
  │ Total GPU blocks:             1982 (block_size=16 tokens)
  │ Max token capacity:           31712
  └──────────────────────────────────────────────────────────────

[you]: hello

[assistant]: Hello! It looks like you're quite friendly today. How can I assist you? Do you have any questions or topics you'd like to discuss?

  ┌─ Request Tokens ─────────────────────────────────────────────
  │ Input tokens (this request):  120
  │ Output tokens (this request): 31
  ├─ Prefix Cache (cumulative, all requests) ────────────────────
  │ Queried tokens:               277.0
  │ Cache hit tokens:             192.0
  │ Cache hit rate:               69.3%
  │ KV cache reads  (from cache): 192.0 tokens
  │ KV cache writes (to cache):   85 tokens
  ├─ KV Cache Pool ──────────────────────────────────────────────
  │ Total GPU blocks:             1982 (block_size=16 tokens)
  │ Max token capacity:           31712
  └──────────────────────────────────────────────────────────────

Note that we are using blocks of 16. Anything that is not inside the full block will not be cached and will be recomputed. Hence, not all generated tokens will be used as prefix cache in the next turn
Note that in turn one, there is the input prompt + system prompt. In the next turn there is previous prompts + assistant prompt + whatever, each turn is not just the user prompt. 

---

Running claude code on the GPU machine

```
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
HOST=0.0.0.0
PORT=8000
TENSOR_PARALLEL_SIZE=2
DTYPE=auto
MAX_MODEL_LEN=65536
GPU_MEMORY_UTILIZATION=0.90
CUDA_VISIBLE_DEVICES=6,7
```

`./launch_vllm.sh`
`./run_claude_local.sh`
`./watch_cache.sh 1`

![Claude Code running against local vLLM server](claude_local.png)