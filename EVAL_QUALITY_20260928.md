# Bonsai 2 27B（PTQ1_0 三元）质量实测报告

日期：2026-09-28
环境：AutoDL RTX PRO 6000 96GB / vLLM 0.26 / 自研三元 kernel（压缩态驻留 5.9GB）
被测服务：`bonsai2-27b`（serve_bonsai.py --from-safetensors --ternary，dtype fp16）
方法：temperature=0，并发 16/12，原始回答全量落盘 `/root/eval/results_ternary/*_raw.jsonl`

## 一、总分对照官方宣称

| 科目 | 样本 | 实测 | 官方宣称 | 判定 |
|---|---|---|---|---|
| MMLU | 342（57学科×6） | **85.96%** | 14 项均分 84.78（FP16 86.32） | ✅ 达标，高于官方均分 |
| GSM8K | 100 | **95.0%** | "数学与全精度差半分以内" | ✅ 强力支持，27B 级优秀 |
| HumanEval | 60 | **70.0% pass@1** | "编程与基线持平" | ⚠️ 良好但大概率略低于 FP16 基线（27B 级 FP16 通常 75-85） |

综合：官方"98.2% 保留率"在数学、综合知识两科得到实测支持；编程不拉胯但"持平"说法偏乐观。
注意：官方 84.78 是 14 项 benchmark 均分，与本文三科不可直接逐项对表。

## 二、评测过程发现（比分数更有价值）

1. **max_tokens 必须给足**：xhigh 思考模式下模型单题可思考上万 token。
   HumanEval 首轮 max_tokens=4096 时 58/60 答案没写完（pass@1 仅 3.3%），
   提到 10240 后修复至 70.0%。生产部署建议 max_tokens ≥ 8192。
2. MMLU 首轮 12 题截断（1500/2500 上限），修复后 84.21%→85.96%。
3. GSM8K 100 题零截断——数学题的思考链反而收敛。
4. 仍有 3 道 HumanEval 在 10240 处截断判负（HumanEval/116, /145, /32），
   真实 pass@1 可能再高 1-3 个百分点。

## 三、MMLU 分学科短板（修复前口径，57 学科全表见 mmlu_summary.json）

- 满分学科（25 个）：数学类（abstract_algebra、college_mathematics、elementary/high_school_mathematics、
  college_physics、formal_logic、machine_learning、electrical_engineering 等）——推理密集型无损。
- 明显短板：global_facts 50%、professional_law 50%、public_relations 50%、
  college_chemistry 50%——事实记忆/法规/冷知识类，符合低 bit 量化典型退化面。

## 四、复现方式

```bash
# 服务器上（服务需先启动：bash /root/start_vllm_ternary.sh）
cd /root/eval
python run_eval.py /root/eval/results_ternary mmlu gsm8k humaneval
python repair_eval.py /root/eval/results_ternary   # 截断题修复
```

数据集：/root/eval/data/sample/（MMLU 342 / GSM8K 100 / HumanEval 60，seed=7 可复现；
全量备份在 /root/eval/data/）。

---

## 五、FP8 生产基线对照（同口径实测，2026-09-28 晚）

基线：Qwen 官方 Qwen3.8-27B-FP8（27GB，ModelScope 下载），stock vLLM 0.26 服务；
与三元评测同题集、同脚本、同 temperature=0、同修复流程，唯一变量为权重格式。

| 科目 | 三元 5.9GB | FP8 27GB | 差距 | 保留率 |
|---|---|---|---|---|
| MMLU（342题） | 85.96% | 90.35% | -4.4 pt | 95.1% |
| GSM8K（100题） | 95.0% | 96.0% | -1.0 pt | 99.0% |
| HumanEval（60题） | 70.0% | 75.0% | -5.0 pt | 93.3% |

速度对照：MMLU 评测耗时 三元 1057s vs FP8 264s（FP8 快 4 倍）；GSM8K 278s vs 134s（2 倍）。

结论修正（配对 McNemar 精确检验，同题集逐项对比）：
1. MMLU：FP8 显著优于三元（仅FP8对21题 vs 仅三元对6题，p=0.0059），-4.4 pt 为真实差距，
   知识类存在可测退化；
2. GSM8K：两者无显著差异（p=1.0），"数学与基线基本持平"成立；
3. HumanEval：-5.0 pt 差距统计不显著（仅FP8对3题 vs 仅三元对0题，p=0.25，60题样本功效不足），
   本数据既不能证实也不能证伪官方"编程持平"说法；且官方未公开评测口径
   （benchmark/采样/pass@k 均未知），若要判定需扩样至全量164题或使用官方同款benchmark；
4. 官方"98.2% 保留率"：本实测三科保留率 95.1/99.0/93.3，方向一致、同量级，
   但精确数值受评测口径影响，不宜直接对表；
5. 速度维度：FP8 硬件加速带来 2-4 倍吞吐优势，中心集群选 FP8 无争议；
   三元生态位为边缘/端侧/显存硬约束场景。
