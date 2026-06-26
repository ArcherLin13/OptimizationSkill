# Flash Attention

## 一句话

**不物化完整 n×n 注意力矩阵**；分块在 SRAM 里算，结果与标准 Attention 一致（exact），显存从 O(n²) 降到 O(n) 量级。

## 标准 Attention 的问题

```text
S = QK^T   →  n×n 写入 HBM  （峰值大户）
P = softmax(S)  →   again n×n
O = P × V
```

n=32K 时，单矩阵可达 GB 级。

## Flash 做法

- **Tiling**：Q/K/V 分块
- **Online softmax**：不看完所有块也能等价 softmax
- **Fusion**：少写中间结果
- **Recompute**：反向时不存 P，用算力换 HBM

## 与 Camera 内存优化的类比

| Camera | Flash Attention |
|--------|-----------------|
| 避免每帧 malloc 大 buffer | 避免物化 n×n |
| memory pool 钉峰值 | tiling 钉峰值 |
| 在本地池复用 | 在 SRAM 分块算 |

## 关联

- [[attention-qkv]]
- [[llm-basics]]
