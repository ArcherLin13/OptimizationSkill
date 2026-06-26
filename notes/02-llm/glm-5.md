# GLM-5.1 / 5.2 部署

## 一句话

**744B MoE**，FP8 权重约 **744～750 GB**；生产需 **8× H200（141GB）** 单机 FP8，普通 PC / 单卡 4090 **无法部署**。

## 规格

| | 5.1 | 5.2 |
|---|-----|-----|
| 总参数 | ~744B MoE | ~744B MoE |
| 每 token 激活 | ~40B | ~40B |
| 上下文 | ~200K | **~1M** |
| FP8 权重 | ~860GB 量级 | ~744GB |

## 推荐生产配置

```text
GPU:    8× H200 141GB（或 8× H20 / B200 冲 1M ctx）
权重:   zai-org/GLM-5.x-FP8
框架:   vLLM ≥ 0.23 / SGLang
并行:   --tensor-parallel-size 8
建议:   --kv-cache-dtype fp8
```

| 组件 | 建议 |
|------|------|
| 显存合计 | ~1128 GB（8×141） |
| 系统内存 | 256～512 GB |
| 磁盘 | ~750 GB（FP8）～1.5 TB（BF16） |

## 替代

- API（智谱 / OpenRouter）
- 自托管 **GLM-4.7**（约 4× H100/H200）
- 本机 **GLM-4-9B / 32B** 量化

## 启动示例（5.2）

```bash
vllm serve zai-org/GLM-5.2-FP8 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice
```

## 关联

- [[deployment-hardware]]
- [[../03-hardware/gpu-pricing]]
