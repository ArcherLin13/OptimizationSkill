# 新优化 bench 清单

复制到新 bench 目录，逐项打勾。

## 立项

- [ ] 优化目标函数/模块名：________________
- [ ] 输入尺寸/格式：________________
- [ ] 正确性标准：SAME(maxDiff=0) / FAST(PSNR___ maxDiff___) / APRX
- [ ] 目标 speedup：___× vs baseline

## 脚手架

- [ ] `init_bench.ps1 -Name ___` 已执行
- [ ] `config.local.ps1` 已配置 `OHOS_NATIVE`、`REMOTE_DIR`
- [ ] `lib/your_optimized/README.md` 已填写合回说明

## 实现

- [ ] `baseline` = 大项目当前行为
- [ ] `bench_case` 有 default + samples
- [ ] `benchmark.cpp` 注册所有 variant
- [ ] 优化代码只在 `lib/`，bench 无业务逻辑

## 设备验证

- [ ] `build.ps1` 通过
- [ ] `deploy.ps1` 真机可执行
- [ ] SAME 路径 PASS
- [ ] 耗时记录（avg/min/max）已存档
- [ ] （可选）`pull_results.ps1` 拉回 diff 图

## 合回大项目

- [ ] `lib/` 已拷贝到主工程
- [ ] 主工程链接依赖已更新
- [ ] 主工程全链路冒烟
- [ ] git tag：`___-pass`
