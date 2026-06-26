# 大模型在干什么

## 一句话

LLM = 超大规模的「下一个词预测器」：输入文本 → 多层 Transformer → 输出下一个 token 的概率。

## 流程

```text
文本 → Tokenize → Embedding
     → ×N 层 Transformer（Attention + FFN）
     → logits → 采样下一个词
```

## 组件

| 组件 | 作用 |
|------|------|
| **Attention** | 词与词之间「看谁」 |
| **FFN** | 每个词非线性变换 |
| **Multi-Head** | 多套 Q/K/V 并行 |
| **KV Cache** | 推理时缓存历史 K/V，省重复计算 |

## 关联

- [[attention-qkv]]
- [[flash-attention]]
