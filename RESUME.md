# 简历素材

## 一段话版

独立完成三元量化大模型 Bonsai 2 27B 的 vLLM 移植与生产级优化：在无文档条件下逆向私有量化格式（base-3 三值打包 + Hadamard 旋转基底），通过"参考实现全图算子 dump + 逐层对账"定位并修复 6 个深层数值 bug，输出与官方实现逐字对齐 98.94%（fp16 参考路径 99.79%）；自研 PTQ1_0 三值解码 Triton kernel（实测带宽 1.3TB/s），权重以 5.9GB 压缩形态常驻显存（免 54GB fp16 展开），并解决 dynamo/CUDA Graph 编译兼容问题完整接入 vLLM 编译管线；最终实现单流 90.6 tok/s（较 fp16 基线 3.4 倍）、8 并发 204 tok/s，反超模型官方 llama.cpp 实现 34%（官方 server 默认配置口径）。

## 精简版（一句话）

独立完成三元量化大模型（Bonsai 2 27B）的 vLLM 移植：逆向私有量化格式，自研 Triton 三元解码 kernel 并接入 CUDA Graph 编译管线，生产并发负载下吞吐 204 tok/s，超越模型官方实现 34%（官方 server 默认配置口径），显存占用 54GB→5.9GB。

## 关键词（用于简历技能栏/检索匹配）

vLLM、Triton kernel、CUDA Graph、量化推理（Ternary/PTQ）、GGUF、llama.cpp、数值对齐、推理性能优化、Hadamard、大模型部署
