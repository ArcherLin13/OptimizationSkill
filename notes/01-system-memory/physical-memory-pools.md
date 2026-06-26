# 物理内存池（system / CMA / carveout）

## 一句话

SoC 把 DRAM 切成多种池；Camera/GPU 通过 dma_heap 从对应池分配 buffer。

## 常见池

| 池 | 连续？ | 谁用 | 峰值特点 |
|----|--------|------|----------|
| **system / buddy** | 不保证 | malloc、部分 DMA | 随应用波动 |
| **CMA** | 保证 | Camera、Video、Display | 池满即失败，与 MemFree 无关 |
| **carveout** | 固定预留 | GPU/DSP/Display 固件 | 永远占用，Linux 常不可见 |
| **secure** | TZ 保护 | DRM、TEE | 独立统计 |

## 普通 App 用哪些

| App 类型 | 用的池 |
|----------|--------|
| 纯业务（网络、JSON） |  mainly **buddy**，一般不碰 CMA |
| 带 UI / 相机 / 视频 | 框架底层 **dma-buf**，常来自 **CMA + system** |

## 怎么查（Android / Linux）

```bash
ls /dev/dma_heap/
cat /proc/meminfo | grep -i cma
cat /d/dma_buf/bufinfo    # 部分机型
adb shell dumpsys meminfo
```

## 关联

- [[ion-dma-buf-dma-heap]]
- [[camera-memory-pool]]
