# Softmax

## 一句话

把一串实数变成 **和为 1 的正权重**；在 Attention 里把相似度变成注意力分配。

## 公式

```text
softmax(z_i) = exp(z_i) / Σ exp(z_j)
```

## 为何叫 soft**max**

- **max**：最大的项拿到最多权重
- **soft**：不是 0/1 硬选，而是平滑概率，且 **可微** 便于训练

## 在 Attention 中的位置

```text
QK^T / √d  →  softmax  →  × V
   分数         权重       输出
```

## 关联

- [[attention-qkv]]
