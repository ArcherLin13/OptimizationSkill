# ION、dma_heap、dma-buf

## 一句话

**dma-buf** 是内核里可跨设备共享的 DMA 内存（fd）；**dma_heap / ion** 是分配它的入口，不是另一块物理内存。

## 层级

```text
应用 / HAL / Camera / GPU
        ↓ open + ioctl
/dev/dma_heap/*（新）  或  /dev/ion（旧）
        ↓
struct dma_buf → 返回 fd
        ↓
同一块物理 RAM（system / CMA / carveout …）
```

## 对比

| | dma-buf | ion | dma_heap |
|---|---------|-----|----------|
| 是什么 | 内核共享对象 + fd | 旧分配框架 | 新分配入口（每池一设备文件） |
| 关系 | 产物 | 分配方式之一 | 分配方式之一（现行标准） |

## `/dev/ion` 和 `/dev/dma_heap` 为何都在

迁移期并存：老 HAL 走 ion，新栈走 dma_heap，底层常是同一物理池。

## 设备文件

`/dev/ion`、`/dev/dma_heap/cma` 等是 **字符设备文件**，不是存数据的普通文件；`open` 后由内核驱动响应 `ioctl` 分配内存。

## 共享谁保证

- 硬件前提：SoC 上 CPU/GPU/Camera 连同一 DRAM
- 软件保证：**dma-buf 子系统 + 驱动 attach + IOMMU + cache sync**
- fd = 同一块 `dma_buf` 的凭证，可传给多设备

## 易混点

- **system RAM（buddy）** ≠ **`/dev/dma_heap/system`**：前者给 malloc，后者给 DMA 共享
- MoE「40B 激活」≠ 只加载 40B 权重（部署话题，见 [[../02-llm/glm-5]]）

## 关联

- [[physical-memory-pools]]
- [[camera-memory-pool]]
