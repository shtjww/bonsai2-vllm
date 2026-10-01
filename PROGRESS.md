# Bonsai2-vLLM 项目进展日志 — 2026-09-23

## 09-24 上午：克隆新实例状态

- 新实例：RTX PRO 6000 Blackwell 96GB（<autodl-instance>），克隆模式数据完整
- **容器有 110GiB 内存 cgroup 上限**（free 显示的 1TB 是宿主机的！）——fp32 全量权重 ~100GiB 会被静默 OOM 杀掉，一切中间产物必须 fp16
- **/dev/shm 仅 55GiB、autodl-tmp 仅 44GB 空闲**——缓存放 /dev/shm（RAM 速度），用后即删

## 09-24 中午：根因定位（连环四 bug）

排查路径：fp16/bf16 都乱码排除精度 → HF transformers 用我们的权重输出**与 vLLM 逐 token 相同的乱码** → 锁定权重转换 → fork `llama-eval-callback` dump 全图算子值 → embedding 层就分叉 → 逐角值对比实锤：

1. **【万恶之源】gguf_reader 布局 scramble**：ggml 存储 ne0 最快维，flat 字节流 C-order 解释应该是**反转后的 shape**。代码却 `reshape(t.shape)` 再转置 → 所有 2D 张量数据全乱（同 block 三值权重出现 79 个不同幅度，违反块内共享 scale）。昨天所有 1e-6 验证全是同一份错误数据的自洽验证（循环论证）。修复：`reshape(t.shape[::-1])`，且 2D 权重**不需要转置**（反转后天然就是 torch 的 (out,in)）。embedding 修复后与 fork 角值逐位吻合（-1.583417 vs -1.5834174）
2. **norm 权重约定**：vLLM Qwen3_5 用 GemmaRMSNorm（y=x·(1+w)，零中心化存储），GGUF 存 1 中心化 → loader 里 -1。例外：GDN 的 ssm_norm 是 RMSNormGated（标准 x·w）不动
3. **qkv 重排画蛇添足**：vLLM Qwen3.5 路径（gqa_interleaved_layout=False）要的就是 llama.cpp 原版连续 [q|k|v]；昨天的组交错 repack 是按 Qwen3-Next 的错误假设加的 → 全删（含 conv1d 通道重排）
4. **A_log 双重变换**：GGUF 的 ssm_a 已存 -exp(A_log)（fork qwen35.cpp:459 直接乘），vLLM/HF 会再做一次 -exp() → loader 里 A_log = log(-ssm_a)

## 工具链升级

- `vllm_bonsai/dump.py`：BONSAI_DUMP_DIR 环境变量触发逐层 hidden states dump（自动 enforce-eager）
- `serve_bonsai.py`：新增 --gpu-memory-utilization / --dtype / --from-safetensors（快启 ~2min，跳过 8min GGUF 反量化）
- `cache_to_safetensors.py` + `add_visual.py`：GGUF → fp16 safetensors 一次性导出（/dev/shm）
- `compare_fork.py`：fork eval-callback 算子 sum vs HF 逐层 hidden states 对比
- gguf_reader 新增 `tensor_torch()`（正确朝向）；`tensor_f16()` 保留旧语义并标注警告
- **convert_to_bf16.py / convert_to_safetensors.py 有同样的布局 bug，已废弃勿用**（safetensors 导出走 cache_to_safetensors.py）

5. **ssm_out 置换方向反了**：checkpoint 存储本就是 tiled 顺序（vLLM 要的），fork 的运行时置换是适配它自己 grouped kernel 的，权重里不该动
6. **GDN v 头 GQA 配对**：fork kernel `iq1 = h_idx % 16`（隔行），vLLM/HF fla kernel `i_h // (H//Hg)`（连续）→ loader 里对 v 相关维度（qkv-v/z/conv-v/a/b/A_log/dt_bias）做 grouped→tiled 置换

## 09-24 午后：✅ logits 对齐完成

compare_logits（8 prompt × 64 token greedy）：**top1 一致率 99.79%**，mean |Δlogprob| 0.0025（fp16 噪声级）。
唯一分叉点在某 prompt 第 22 token（边界 logit 翻转，正常）。

服务方式：safetensors fp16 快启（/dev/shm/serve-bonsai2-fp16，~2min；GGUF 路径 ~7min 仍可用）。
safetensors 重建链：`bash /root/full_chain.sh`（GGUF→缓存→safetensors→视觉塔→启动）。

**明天第一步**：fused ternary kernel（显存常驻 5.9GB，kernel 内解码）。

**09-24 压测基线**（bs=1 greedy 256tok，RTX PRO 6000）：vLLM fp16 = **26.6 tok/s**（≈1436GB/s 已打满带宽）；
fork 三元原生 = **107.0 tok/s**。差距 4x = fused kernel 的潜在收益。H800 FP8 基线 205→296 tok/s（不同卡，仅参考）。
注意 /dev/shm 是内存盘：关机前如需保留 safetensors 要拷到 autodl-tmp（或接受 10 分钟重建）。

## 09-24 傍晚：✅ 三元常驻 kernel 上线（vLLM 编译模式）

**59.3 tok/s（2.2x fp16）**，输出与 fork 逐字一致。路径：
- `vllm_bonsai/ternary_kernel.py`：Triton PTQ1_0 GEMV/GEMM（uint32 合并加载 + split-K + 原子加），注册为 torch custom op `bonsai::ternary_hadamard_linear`（对 dynamo 不透明 → 编译兼容）
- `vllm_bonsai/ternary_layers.py`：TernaryHadamardLinear / TernaryLMHead（packed 注册为 buffer；weight 放空 Parameter 占位；bias 返回 dummy 张量而非 None）
- `vllm_bonsai/ternary_swap.py`：加载后把 401 个 folded 张量换成 packed（含 v 头 grouped→tiled 置换），257 个模块替换
- 启动：`serve_bonsai.py --from-safetensors --ternary`（或 /root/start_vllm_ternary.sh）

**编译兼容四坑**（顺序）：① dynamo 查 `_parameters['weight']` → 占位空 Parameter；② forward 返回 tuple 含 None → inductor 不收 None，改 dummy 张量；③ 占位 weight 建在 CPU → 显式 cuda；④ **陈旧 torch_compile_cache 复用了 fp16 版的图** → 改模型结构后必须 `rm -rf /root/.cache/vllm/torch_compile_cache`

**明天：kernel 带宽调优**（436GB/s → 800+ 目标）：uint32x4 向量加载、按张量调 SPLIT_K、lm_head 单独调参 → 摸 100 tok/s。

**09-24 晚：kernel 调参后 73.4 → 最终 90.6 tok/s（3.4x fp16，fork 的 84%）**，输出仍逐字正确。
分档配置（ternary_matmul 内置启发式）：lm_head 级（≥5e8）BO=256/SK=4（1289GB/s）；n_out≥16384（ffn/ssm_out 级）BO=256/SK=8（1111GB/s）；其余 BO=128/SK=16（776GB/s）。
教训：小 tensor 是 kernel 延迟受限（25μs 地板），向量化 x 加载反而慢（寄存器压力），标量 x + split-K 才是正解。
bs=1 已逼近小 kernel 延迟地板；后续空间在批量解码（bs>1 摊薄）或全层融合大 kernel（工程量大）。

**09-24 深夜：批量扩展性收官**（greedy 256tok 聚合吞吐）：
| 并发 | vLLM 三元 | fork |
|---|---|---|
| bs=1 | 90.5 | 108.3 |
| bs=4 | 163.5 | 163.1 |
| bs=8 | **206.1** | 160.1（vLLM 反超 29%）|
小 kernel 延迟被批量摊薄；fork/llama.cpp bs>4 后无扩展性。**真实生产负载下 vLLM 三元版已是最快。**
三元版全量锚点：98.94% top1 一致、Δlogprob 0.0025（split-K 原子加非确定性 + fp16 scale 舍入，正常噪声）。

## 09-25：✅ 视觉模态验证通过

mmproj 转换（patch_embed 双时序片堆叠、merger norm、pos_embed）在布局大修复时已一并治好。
端到端实测（自制测试图：红圆+蓝方块+数字42）：vLLM fp16 版和三元版均与 fork llama-mtmd-cli 描述一致。
测试工具：`/root/test_image_vllm.py`（base64 → OpenAI multimodal API）、`/root/fork_mtmd_test.sh`（fork 参照）。
坑提醒：fp16 ↔ 三元版互切时必须清 /root/.cache/vllm/torch_compile_cache（结构不同，缓存图复用会炸）。

---

# Bonsai2-vLLM 项目进展日志 — 2026-09-23

## 今日成果

### 1. 格式逆向（完成，CPU数值验证 1e-6）

PrismML Ternary Bonsai 2 27B 私有格式完全逆向（源：PrismML-Eng/llama.cpp fork）：

- **PTQ1_0**（ggml type 143）：128权重/28字节 = qs[24](120 trits, 5trits/字节 base-3) + qh[2](8 trits) + fp16 scale → 1.75 bpw。trit提取：`(uint8)(q*3^n)*3>>8`（注意uint8回绕）
- **PQ2_0**（type 142）：128权重/34字节 = fp16 scale + 32B 2bit码（同为三值{-1,0,+1}，更快解码的容器）
- **Hadamard**：不存矩阵。`prism.hadamard.block_size=1024`，运行时生成 `H[r,c]=(-1)^popcount(r&c)/√n`；另有全宽±1符号向量（5120/6144/17408三档，sign_mode=explicit）；GDN ssm_out 先特征置换 [hd,nk,rep]→[hd,rep,nk]
- **正向折叠权重**：`y = W_q^T (H (s ⊙ P(x)))` → 还原 `W_std = P^T diag(s) H W_q`
- **embedding（token_embd）反向**：lookup后 `h = s ⊙ (H z)`，还原公式同形
- **架构**：qwen35，64层 = 48×GDN(linear_attn) + 16×全注意力（interval 4），embd 5120，head_dim 256，GQA 24/4，GDN: k头16×128 + v头48×128
- 三元纯度实测：非零权重精确±d，零值32.8%

### 2. 自研工具链（D:\2026年工作\bonsai2-vllm）

```
src/          gguf_reader / prism_dequant / hadamard / name_map / convert_to_* / download_parallel（断点续传）
vllm_bonsai/  vLLM loader 插件：bonsai_gguf load format，dequant-at-load（GPU加速FWHT）
reference/    官方 Qwen3.8-27B config/tokenizer/index（映射对照来源）
deploy/       remote.py(paramiko驱动) / server_bootstrap / fork_setup / poll / watch
```

- ggml→HF 名称映射：851↔851 双向全覆盖（已用官方index验证）
- 转换数值闭环：attn_qkv/ffn/ssm_out 三类权重 vs fork运行时语义，max rel err ~1e-6

### 3. 服务器环境（AutoDL RTX 6000D 85GB，西北企业区）

- vLLM 0.26.0 装好（uv + tuna镜像；踩坑：vllm钉死torch==2.11.0需重装~6GB；实例pypi限速~1MB/s持续、突发快）
- 权重：PTQ1_0 5.9GB + mmproj 931MB 已就位 `/root/autodl-tmp/models/`
- vLLM服务 :8000（bonsai2-27b）✅ 在线；fork llama-server :8080 ✅ 在线且**输出正确**（"Paris"✓）
- 学术加速：`source /etc/network_turbo` 只对HF/GitHub快，对pypi更慢（分开用）

### 4. 关键bug发现（博客素材）

**llama.cpp vs vLLM 的 GDN qkv 打包顺序不同**：
- llama.cpp：连续 `[q(2048), k(2048), v(6144)]`（按offset切分）
- vLLM：组交错 `[(q_g,k_g) × 16组, v]`（fused qkvz view后按组split）
- 已在loader加 `_repack_gdn_qkv`（含conv1d通道同步重排）
- 全注意力QG打包（[q_h,gate_h]头交错）两边一致，无需处理；b/a、z 两边一致

### 5. 已修的形状/兼容性坑

- vLLM 0.26 API：`DefaultLoader`→`DefaultModelLoader`；`_get_weights_iterator(source)`；`FlexibleArgumentParser`挪到`vllm.utils.argparse_utils`；`run_server`是async；注册表`_LOAD_FORMAT_TO_MODEL_LOADER`
- 必须提供视觉塔：mmproj 334张量映射（v.blk→visual.blocks；patch_embed双时序片stack→Conv3d；v.post_ln=merger.norm(1152)）
- conv1d：ggml(kernel,channels) → torch(channels,1,kernel)
- Mamba cache blocks上限：max_num_seqs默认1024 > 可用427 → 设256（H800上是820，卡越小越少）

## 当前状态（09-23 深夜）

**三路数值校验全部通过**（服务器 `/root/validate.log`）：
1. embedding inverse: rel err = 0.00e+00 ✅
2. lm_head forward: rel err = 1.57e-06 ✅
3. qkv repack: q/k/v ≈ 1e-06 ✅（修正后）

**关键修正**：qkv重排时 **v也必须按组交错**——vLLM期望每组 [q_g(128), k_g(128), v_g(384)]，
不是v整块拖尾。已写入 `vllm_bonsai/weights.py::_repack_gdn_qkv`。

**服务器上跑着的V14服务还是旧（错）权重**，明天必须重启。

## 明天第一步（按序）

1. 重启vLLM（V15，带修正后的repack）→ 发同样3个prompt → 输出应对齐fork
2. 对齐后跑 src/compare_logits.py（logprobs逐token对比）→ **正确性锚点**
3. run_bench.sh 同口径压测 → 对比FP8 H800基线（205→296 tok/s）
4. 若仍乱码，候选：A_log约定、conv_state dtype、RoPE sections

## 服务器备忘

- 实例：RTX 6000D（<autodl-instance>），console创建
- 权重/代码/双服务配置全部就位，重启服务命令：
  `export PATH=/root/miniconda3/bin:$PATH PYTHONPATH=/root/bonsai2-vllm HF_HUB_OFFLINE=1; cd /root/bonsai2-vllm && python vllm_bonsai/serve_bonsai.py --gguf-dir /root/autodl-tmp/models/serve-bonsai2 --port 8000`
- fork服务：`/root/bonsai2-vllm/fork-bin/llama-server -m /root/autodl-tmp/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf --port 8080 -c 8192 --alias bonsai`

## 费用备忘

- 实例在计费中（RTX 6000D），不用时控制台关机（数据保留，开机继续）
- 旧5090 Pro实例已释放 ✅

## 博客素材积累

- 格式逆向全过程（base-3/uint8回绕/Hadamard不存矩阵）
- llama.cpp vs vLLM qkv打包顺序暗坑
- AutoDL API的坑：公共镜像无API、pypi限速、turbo双刃剑、GET+JSON body
- 下载器的坑：xet协议国内死、签名URL过期、双实例撞车、线程异常不传播

---

## 2026-10-01 补录：发布前审查 → v1 落败认账 → v2 负结果 → v3 反杀

- 发布前"敌意审稿"自查出两类问题：(a) deploy 脚本硬编码实例 SSH 明文密码（已清并重写 git 历史）；(b) 全部并发对照里 fork 为默认 4 槽配置，"反超 51%"系排队假象。
- fork 满血（-np 32 -c 65536）重测：bs=8/16/32 = 184.7/299.4/372.8。v1 在 bs≥16 落败（226.0/233.9），如实改 README 双配置口径。
- v2（BLOCK_T 批量，减权重流量 16×）：微观 +44% 系 L2 幻觉（同张量循环），端到端 bs=16 崩盘（126.0 vs 226.0）。负结果封存，BONSAI_KERNEL_V2=1 可复现。
- v3（tensor core）：索引向量化解包 (128,BO) fp16 tile + tl.dot MMA，BT=16/BO=64/SK=8/NW=4/NS=1（BO>64 SMEM 爆）。微观 4.9×/8.0×；端到端验证轮 bs=16 759.3 / bs=32 889.7 = 2.5~2.6× 满血 fork，两轮复现 ≤3% 方差。
- 正确性：32 并发×256tok 对拍 25/32 逐字一致，7 个发散均为确定性后段 fp16 舍入。T=1 路径未动，98.94% parity 结论不变。
- 全天实验细节：BENCH_20261001.md。
