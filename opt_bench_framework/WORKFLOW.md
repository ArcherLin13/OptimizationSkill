# 独立优化测试工作流

## 为什么要剥离

| 大项目直接测 | 独立 bench |
|--------------|------------|
| 编译慢、依赖多 | 只编一个可执行文件 |
| 启动相机/业务链路过长 | 秒级跑完 |
| 难以 A/B 对比 | baseline / optimized 同表输出 |
| 回归困难 | 固定 samples + 阈值 |

## Phase 0：界定范围

1. **一个函数 / 一条热路径**（例如 `seamlessClone`、某个 filter、一次 memcpy）
2. **固定输入形状**（例如 729×126、固定 mask）+ 至少 1 组真实 `samples/`
3. **成功标准**写清楚：
   - 像素一致：`maxDiff=0`
   - 或 PSNR / maxDiff 阈值
   - 目标耗时（相对 baseline 的 speedup）

## Phase 1：脚手架

```powershell
.\opt_bench_framework\scripts\init_bench.ps1 -Name your_bench
```

填写 `template/README.md` 里的 **Optimization brief**。

## Phase 2：实现三层

### 1. `bench_case` — 输入

- `makeDefaultCase()`：合成数据，保证无外部文件也能跑
- `loadCaseFromDir(dir)`：从 `samples/xxx/` 读 bmp/png/bin
- 记录到 benchmark 表：尺寸、参数字段

### 2. `baseline` — 大项目现状

- 尽量**原样复制**大项目里的调用（或链到同一第三方库）
- 标记为 `VariantKind::Reference`

### 3. `lib/your_optimized/` — 优化实现

- **可单独拷贝**的 `.h/.cpp`，不依赖 bench 代码
- bench 里只 `#include` + 调用
- 验证通过后，整个目录拷进大项目

### 4. `benchmark.cpp` — 注册表

```cpp
variants.push_back({"baseline", VariantKind::Reference, runBaseline, ...});
variants.push_back({"optimized_v1", VariantKind::Identical, runOpt, 100.0, 0.0, "note"});
```

## Phase 3：设备闭环

```powershell
.\scripts\build.ps1
.\scripts\deploy.ps1
.\scripts\pull_results.ps1   # 若有 --export ./out
```

迭代循环：

```
改 lib/ → build → deploy → 看表 → 直到 SAME PASS + 够快
```

## Phase 4：合回大项目

1. 复制 `lib/your_optimized/` 到大工程
2. 替换调用点（保留 baseline 宏/开关一周便于回滚）
3. 大项目全量回归（UI、多分辨率、边界 case）

## Phase 5：打 tag

```bash
git tag your-bench-v1-pass -m "729x126 maxDiff=0 avg 100ms"
```

便于以后对比退化。

## 常见坑（seamless clone 实战）

| 坑 | 教训 |
|----|------|
| 换 FFT 库求快 | pocketfft / 自研 DFT 易 FAIL，maxDiff=0 必须跟 OpenCV 同路径 |
| skip/cache | 每帧不同场景不要用 |
| tag 打错 commit | 慢路径 refactor 后性能变 baseline 级 |
| `std::thread` 在 OHOS | 优先 `cv::parallel_for_` |
| 只测合成数据 | 加 `samples/` 真图 case |

## 扩展

- **多 case**：`--case text|visual|prod`
- **可视化**：`--export ./out` + `grid_diff_x8.bmp`
- **多尺寸 sweep**：在 `benchmark.cpp` 里循环 `sizes[]`
