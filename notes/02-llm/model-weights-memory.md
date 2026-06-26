# 权重在内存还是显存

## 一句话

**GPU 推理且显存够 → 权重整份在显存**；CPU 推理 → 在 RAM；不够则 offload，能跑但慢。

## GPU 推理（显存够）

```text
硬盘 → 加载 → 显存（权重 + KV + 开销）→ GPU 算
```

系统 RAM 只做加载缓冲、预处理，几 GB 即可。

## 显存占用

```text
总显存 ≈ 权重 + KV Cache + 框架开销
```

## CPU / offload

| 情况 | 权重位置 |
|------|----------|
| 纯 CPU | RAM |
| 显存不够 | 部分在 RAM，按层拷入 GPU |
| mmap | 按需从磁盘进 RAM |

## 关联

- [[deployment-hardware]]
- [[../01-system-memory/ram-vram-storage]]
