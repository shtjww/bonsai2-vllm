# 服务器关机/恢复备忘（2026-09-24 验证版）

## 关机后什么丢、什么不丢

| 内容 | 位置 | 关机后 |
|---|---|---|
| 全部代码（修复+kernel+脚本） | /root + 本地 D 盘 | ✅ 保留（双份，md5 已核对一致） |
| GGUF 模型权重 | /root/autodl-tmp | ✅ 保留 |
| 文档（PROGRESS.md / BLOG.md / PR_SUMMARY.md） | 本地 D 盘 | ✅ 保留 |
| fp16 safetensors（/dev/shm/serve-bonsai2-fp16） | /dev/shm 内存盘 | ❌ 丢，10 分钟可重建 |
| 容器内存 cgroup 上限 | 110GiB（free 显示的 1TB 是宿主机的） | 注意 |

## 下次开机恢复现场（约 10 分钟）

```bash
# 1. 重建 safetensors 并启动 vLLM（GGUF→缓存→safetensors→视觉塔→启动，全自动）
bash /root/full_chain.sh

# 2. 等 vLLM 就绪（~2 分钟）后，切到三元 kernel 版（默认即生产形态）：
bash /root/start_vllm_ternary.sh

# 3. fork 参照服务（按需）
nohup /root/bonsai2-vllm/fork-bin/llama-server \
  -m /root/autodl-tmp/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --port 8080 -c 8192 --alias bonsai > /root/fork-server.log 2>&1 &
```

## 验证命令

```bash
bash /root/test_both.sh        # 双服务输出对比
python /root/run_bench.py      # 单流吞吐
python /root/run_bench_batch.py both   # 并发扩展性
```

## 服务形态

- **vLLM 三元 kernel 版（生产形态）**：`start_vllm_ternary.sh`，:8000
  - 单流 90.6 tok/s，8 并发 204 tok/s，显存常驻 ~5.9GB packed
- vLLM fp16 展开版（调试基线）：`start_vllm_fast.sh`，26.6 tok/s
- fork llama-server（正确性参照，默认 4 并发槽配置）：:8080，单流 99.3 tok/s

## 实例信息

- RTX PRO 6000 Blackwell 96GB，ssh -p <port> root@<autodl-instance>
- vLLM 0.26.0 / torch 2.11.0 / transformers 5.17.0 / Python 3.12（/root/miniconda3）
- **改模型结构后必须 `rm -rf /root/.cache/vllm/torch_compile_cache`**（否则复用旧编译图）
