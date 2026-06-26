# Camera 算法内存池

## 一句话

在算法层预预留 pool，把 **峰值和分配延迟** 钉死在已知上限；与 HAL/CMA 是不同层级的池化。

## 三层池

```text
算法 memory pool     ← 你们预留的 slot / arena
Framework buffer 池  ← 3-buffer 等
CMA / dma_heap       ← 物理池
```

## 为何要 pool

- 多阶段大 buffer，若每步 new/delete → 峰值高、延迟抖动
- 实时 pipeline（30fps）要避免运行时 alloc

## 常见设计

| 类型 | 适用 |
|------|------|
| 固定 block pool | 同尺寸帧缓冲 |
| 分档 tier pool | 多种临时 buffer |
| Ring / pipeline 复用 | 串行 DAG，峰值最低 |
| Arena（每帧 reset） | 帧内大量小对象 |

## 峰值估算

```text
Peak_algo ≈ Σ(同时 live 的 buffer)
Peak_total ≈ HAL + algo_pool + display + …
```

要考虑：**stage 并行度**、**in-flight 帧数**（双缓冲 ×2）。

## 优化顺序

1. 去全量加载 / 临时大 vector
2. 并行局部 buffer（× 线程数）
3. 算法空间复杂度（滚动数组等）
4. backing 尽量统一 dma-buf，避免 CPU↔设备拷贝

## 关联

- [[physical-memory-pools]]
- [[../agent-memory-peak]]（待写：性能优化 Agent 方向）
