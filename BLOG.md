# 把一个三元量化模型塞进 vLLM：逆向、对账与 kernel 优化全记录

> 两天时间，从零文档的私有量化格式，到 vLLM 上 204 tok/s（8 并发）——比模型官方实现（默认 server 配置）快 34%。
> 这是一篇完整的踩坑实录：6 个连环 bug、一次教科书级的"输出乱码"排查、一个 Triton 三元 kernel、以及 vLLM 编译器的四个深坑。

## 零、这是什么

**Bonsai 2 27B** 是 PrismML 的一个三元量化大模型：270 亿参数，每个权重只取 {-1, 0, +1} 三个值，整个模型打包后只有 **5.9GB**（fp16 要 54GB）。

代价是：官方只给了一个 llama.cpp fork 能跑它。而生产环境要的是 **vLLM**——连续批处理、并发扩展、部署生态。

这篇文章记录我把它移植到 vLLM 的全过程。最终成绩：

| 场景 | 吞吐 | 对比 |
|---|---|---|
| vLLM 朴素移植（fp16 展开） | 26.6 tok/s | 带宽打满，不可用 |
| **vLLM + 自研三元 kernel（单流）** | **90.6 tok/s** | 3.4x |
| 官方 llama.cpp fork（单流） | 108 tok/s | 参照系 |
| **vLLM + 三元 kernel（8 并发）** | **204 tok/s** | **反超官方 34%**（默认配置口径） |
| 官方 fork（8 并发） | 160 tok/s | 并发上不去 |

先放结论：**单流打平官方，并发反超**。文章后面解释为什么并发是关键。

## 一、格式逆向：无文档的二进制考古

模型以 GGUF 格式发布，但量化类型是私有的（ggml type 142/143），官方文档为零。逆向来源是 PrismML 的 llama.cpp fork 源码。

挖出来的结构：

**PTQ1_0 打包**（type 143）：128 个权重压进 28 字节 = 1.75 bit/权重。
- 24 字节存 120 个 trit（三值），base-3 编码，5 个 trit 一个字节
- 2 字节存剩下 8 个 trit
- 1 个 fp16 scale

trit 提取公式有个坑：`((uint8)(q * 3^n) * 3) >> 8`——**注意那个 uint8 回绕**，C 代码里的隐式截断，漏了它全部解码都是错的。

**Hadamard 旋转**：权重不存原始值，存的是旋转后的基底。运行时对 activation 做 `x → H(s·x)` 再做 matmul：
- H：normalized Sylvester-Walsh-Hadamard，block=1024，运行时按 `(-1)^popcount(r&c)/√n` 生成，不存矩阵
- s：全宽 ±1 符号向量（5120/6144/17408 三档），存在 metadata 里
- embedding 是反向的：lookup 之后 `h = s·(Hz)`

**架构**：qwen35——64 层 = 48×GDN（线性注意力）+ 16×全注意力（每 4 层一个），embd 5120，head_dim 256。

到这里，转标准基底的数学是清楚的：`W_std = diag(s)·H·W_q`（沿输入维）。昨晚我把转换管线写完，CPU 上验证 rel err ~1e-6，三路数值闭环全绿。

**然后今天上卡，vLLM 输出了乱码。**

## 二、乱码排查：教科书级"对账"方法论

vLLM 起来了，输出是八国语言乱码。fork 输出正常。compare 脚本显示：**8 个 prompt 全部在第 0 个 token 就分叉**——问题在最前端。

### 2.1 为什么之前的验证全是假绿

先自我批判：昨晚的"1e-6 验证"全是**拿同一份转换代码的输出互相比**——如果读取就有系统性错误，所有验证都是自洽的废话。这是今天最大的教训：

> **自洽性验证 ≠ 正确性验证。正确性锚点必须来自独立的参考实现。**

### 2.2 建立真正的锚点

llama.cpp 有个宝藏工具 `llama-eval-callback`：能 dump 出计算图上**每个算子的实际数值**（角落值 + sum）。用它跑 fork，拿到 embedding、每层 norm、qkv 投影、GDN 各阶段的真值。

然后写一个纯 numpy 的第 0 层前向追踪，逐算子对账：

```
model.input_embed       ours=-1.5834   fork=-1.5834   rel=2.3e-07 OK
attn_norm-0             ours=-98.7025  fork=-98.7027  rel=1.9e-06 OK
linear_attn_qkv_mixed-0 ours=-1384.33  fork=-1385.22  rel=6.5e-04 OK
alpha-0 / beta-0 / z-0  ...            OK
conv_output / l2norm    ...            OK
attn_output-0           ours=44.53     fork=62.99     rel=0.29     BAD ← 这里！
```

**第一个对不上的算子就是病灶**，不用猜。

### 2.3 挖出的 6 个连环 bug

**Bug 1（万恶之源）：GGUF 布局 scramble。**
ggml 张量是 ne0 最快维存储，flat 字节流的 C-order 解释应该是**反转后的 shape**。我的 reader 是 `reshape(t.shape)` 再转置——所有 2D 权重全部错位。证据：同一 block 的三值权重理论上共享一个 scale（非零幅度必须相等），错位后一个 block 里出现 79 个不同幅度。修复后 embedding 和 fork 逐位吻合（-1.583417 vs -1.5834174）。

**Bug 2：norm 权重约定差 1。**
vLLM 的 Qwen3_5 用 GemmaRMSNorm：`y = x·(1+w)`，权重零中心化存储；GGUF 存的是 1 中心化（llama.cpp 直接 `x·w`）。转换时每个 norm 减 1。注意 GDN 的 ssm_norm 是标准约定，不能减——**同一个模型里两种 norm 约定并存**，不看源码不可能知道。

**Bug 3：qkv 多余的重排。**
前一晚我以为 vLLM 的 GDN 要 Qwen3-Next 式的组交错布局，加了 repack。实际上 vLLM 0.26 的 Qwen3.5 路径（`gqa_interleaved_layout=False`）要的就是 llama.cpp 原版连续 `[q|k|v]`。**昨天修的"暗坑"，方向是反的。**

**Bug 4：A_log 双重变换。**
GGUF 的 `ssm_a` 存的已经是 `-exp(A_log)`（fork 直接乘），vLLM/HF 会再做一次 `-exp()`。需要 `A_log = log(-ssm_a)` 转回来。

**Bug 5：ssm_out 特征置换方向反了。**
checkpoint 本来就是 vLLM 要的 tiled 顺序；fork 的运行时置换是适配它自己 grouped kernel 的，权重里根本不该动。

**Bug 6（最隐蔽）：GDN 的 GQA 头配对顺序。**
fork 的 CUDA kernel 里 v 头 j 配 k 头 `j % 16`（隔行配对）；vLLM/HF 的 fla kernel 是 `j // 3`（连续配对）。48 个 v 头、16 个 k 头，两种配对方式下权重语义完全不同。转换时要把所有 v 相关维度（qkv 的 v 段、gate、conv1d 通道、α/β、A_log、dt_bias）做 grouped→tiled 置换。**这个 bug 不读两边 kernel 源码根本发现不了。**

修完 6 个，compare_logits：**8 prompt × 64 token，top1 一致率 99.79%，Δlogprob 0.0025**。收工。

## 三、三元 kernel：带宽才是正义

对齐之后压测：vLLM fp16 版 26.6 tok/s。算一下：54GB 权重 × 26.6 ≈ 1436 GB/s——**已经打满 RTX PRO 6000 的显存带宽**。这就是必须做 kernel 内解码的原因：权重保持 5.9GB 打包常驻，每 token 只读 1/9 的字节。

### 3.1 kernel 设计

数学上有个美妙的性质：folded 权重的计算可以写成 `y = W_q^T·(H(s·x))`——**Hadamard 只作用在 activation 上**（FWHT 或缓存 H 的 batched GEMM，都很便宜），权重全程保持打包。

Triton kernel 的核心是 base-3 解码。踩的坑：
- Triton 的 `arange` 只接受 2 的幂 → 28 字节按 7 个 uint32 字加载
- Triton 张量不支持切片/标量索引 → 编译期 Python 列表分发
- **一个字节映射 bug**：stage c=8 的偏移是 `4wi+j` 我写成了 `8wi+j`——单测方法：one-hot 向量逐位置扫，哪个位置错一眼可见

布局上把打包数据转成 `(n_blk, 7, n_out)` 字主序——out 维连续，每次加载完全合并（coalesced）。

### 3.2 接入 vLLM 编译管线的四个坑

 eager 模式跑通后只有 15.5 tok/s——Python/launch 开销吃掉了一切。**必须进 vLLM 的 torch.compile + CUDA Graph 管线**，连续四个坑：

1. dynamo 描自定义模块时查 `_parameters['weight']` → 注册空占位 Parameter
2. forward 返回 tuple 里的 None → inductor 运行时不收 None → 返回 dummy 张量
3. 占位参数建在了 CPU 上 → 显式 cuda
4. **最骚的：vLLM 的 torch_compile_cache 把 fp16 版的编译产物复用到了三元版上**——编译图里 `extern_kernels.mm` 直接引用已经不存在的 fp16 权重地址。清缓存解决

正道是把数学包成 torch custom op（对 dynamo 不透明），剩下的让编译器自己玩。

### 3.3 调参实录

- 大张量（lm_head 278MB）：BO=256/SK=4，**1289 GB/s**
- 中张量（ffn 类）：BO=256/SK=8，1111 GB/s
- 小张量（qkv 类 11MB）：BO=128/SK=16，776 GB/s——**小 kernel 是延迟受限的**（25μs 地板），不是带宽受限
- 反直觉：把标量 x 加载改成向量化反而更慢（寄存器压力）。标量 + split-K 才是正解

速度爬升：26.6 → 59.3 → 73.4 → **90.6 tok/s**。

## 四、为什么并发反超官方

单流：我们 90.6 vs fork 99.3——fork 略快，因为它的 GEMV kernel 更成熟。

并发才是生产真相（fork 侧为官方 demo 文档的默认 server 配置）：

| 并发 | vLLM 三元 | fork |
|---|---|---|
| 1 | 89.1 | 99.3 |
| 4 | 161.9 | 161.5 |
| 8 | **204.1** | 152.6 |

机制很清楚：vLLM 连续批处理下，**一次权重读取服务 N 个 token**——小 kernel 的固定延迟被摊薄，带宽利用率线性上升；而 fork 的 llama-server 默认只有 4 个并发槽，bs>4 后请求排队、吞吐封顶。

> 2026-10-01 二补：fork 开 `-np 32 -c 65536`（对等槽位）后，bs=8/16/32 为 184.7 / 299.4 / 372.8 tok/s——v1 kernel 在 bs≥16 落败；当天换用 tensor core 的 v3 kernel 后，验证轮 bs=16/32 达到 **759.3 / 889.7 tok/s**，反超 fork 2.5~2.6 倍。完整数据见 README 与 BENCH_20261001.md。

**单流打平、并发反超 34%（默认配置口径）——这个模型在 vLLM 生态里站住了。**

## 五、方法论沉淀

1. **自洽验证不等于正确验证**。正确性锚点必须来自独立实现（fork 的 eval-callback 全图 dump 是宝藏）。
2. **逐算子对账 > 猜**。第一个对不上的算子就是病灶，一次定位一个准。
3. **先算带宽账再写 kernel**。26.6 tok/s × 54GB = 打满带宽，不用跑都知道 fp16 展开没有未来。
4. **编译缓存是状态**。模型结构变了一定要清 torch_compile_cache——这个坑值一下午。
5. **kernel 调参要分尺寸**。大张量带宽优先、小张量延迟优先，一套参数打天下是不存在的。

## 六、补记：一天之内的打脸与反杀（2026-10-01）

发布前做“敌意审稿”，抓出一个致命问题：之前所有并发对照里，fork 的 llama-server 都是用默认配置起的——只有 4 个并发槽。bs=32 时人家 28 个请求在排队，“反超 51%”是排队排出来的假象。

**上午：认账。** fork 开 `-np 32 -c 65536` 满血重测：bs=16/32 分别是 299 / 373 tok/s——v1 kernel 实打实输了。数值记档，README 改成双配置口径。

**下午：走错路。** 假设是“v1 每个 token 重复读权重”的锅，写了 BLOCK_T 批量版 v2，把权重读取降 16 倍。微观压测赢 44%，端到端反而输 44%（126 vs 226 tok/s @bs=16）。复盘才明白：微观循环同一批张量，权重住在 L2 里，减流量优势是幻觉；而 v1 的 32 个 token 程序并发读同一块 20MB 权重时 L2 命中率极高（有效带宽 1.6TB/s）。v2 计算受限（有效带宽只有 250GB/s），输得不冤。

**傍晚：反杀。** 算了一笔账：fork 的 373 tok/s 反推等效带宽 3.1TB/s——DRAM 峰值才 1.4TB/s，唯一的解释是它的矩阵乘打在 tensor core 上。那就别在带宽上缠斗了，写 v3：kernel 内用索引向量化解包把 128 个权重直接展开成 (128, BLOCK_OUT) 的 fp16 tile（CUDA core 干），`tl.dot` 上 tensor core（MMA 单元干），两种硬件单元流水并行。

结果（满血 fork 对等槽位，背靠背验证轮）：

| 并发 | v1 | fork | **v3** | 倍数 |
|---|---|---|---|---|
| 16 | 226 | 293 | **759** | 2.6× |
| 32 | 234 | 355 | **890** | 2.5× |

正确性：32 并发×256 token 对拍，25/32 与单流逐字一致，7 个发散全是确定性后段数值舍入（fp16 MMA 的批量数值噪声，与原版 vLLM fp16 同类）。

这一天留下的教训比胜利值钱：**所有“快”都要先问是不是对照组残血、是不是缓存幻觉；承认被打脸的速度，决定反杀的速度。**

## 尾声

两天，从一段无文档的二进制格式，到 vLLM 上比官方更快的生产级服务。所有代码、6 个 bug 的根因链、kernel 实现、调参数据都在项目里。

三元/超低比特量化正在升温（BitNet 谱系），但"能跑 llama.cpp"和"能进生产"之间隔着整个推理引擎工程。希望这篇记录能帮后来人少踩几个坑。

---

*作者注：文中数字均可复现，压测脚本与原始日志正陆续整理进仓库（部分已在 deploy/）。模型：PrismML Ternary-Bonsai-2-27B；硬件：RTX PRO 6000 Blackwell 96GB；vLLM 0.26.0；对比基准：PrismML 官方 llama.cpp fork（server 默认配置）。*
