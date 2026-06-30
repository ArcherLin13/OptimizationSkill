# __BENCH_NAME__

> 填写你的优化简介（从 WORKFLOW Phase 0 复制过来）

## Optimization brief

- **目标**：TODO（例如 `cv::GaussianBlur` 5×5 on 720×128）
- **Baseline**：TODO（大项目里的实现）
- **正确性**：SAME maxDiff=0 / FAST PSNR___ / APRX
- **目标 speedup**：TODO

## 快速开始

```powershell
copy scripts\config.local.ps1.example scripts\config.local.ps1
.\scripts\build.ps1
.\scripts\deploy.ps1
```

## 你要改的文件

| 文件 | 作用 |
|------|------|
| `src/bench_case.*` | 输入数据 |
| `src/baseline.*` | 大项目现状 |
| `lib/your_optimized/*` | 优化实现（合回主工程） |
| `src/benchmark.cpp` | variant 注册表 |

## 合回大项目

拷贝 `lib/your_optimized/` 到主工程，按该目录 `README.md` 集成。
