# opt_bench_framework

从大项目中**剥离单点优化**、在手机上**快速闭环**的模板框架。

灵感来自 `seamless_clone_bench/`：不用跑整个 App，只测 baseline vs optimized 的**耗时 + 正确性**，测通后再把 `lib/` 拷回大项目。

## 目录

| 路径 | 说明 |
|------|------|
| [WORKFLOW.md](WORKFLOW.md) | 方法论：怎么剥离、怎么测、怎么合回主工程 |
| [CHECKLIST.md](CHECKLIST.md) | 每次开新优化的一页清单 |
| [template/](template/) | **复制即用**的 bench 骨架（CMake + OHOS 脚本 + 示例代码） |
| [scripts/init_bench.ps1](scripts/init_bench.ps1) | 从 template 生成新 bench 目录 |

## 5 分钟开新 bench

```powershell
cd opt_bench_framework
.\scripts\init_bench.ps1 -Name my_kernel_bench
cd ..\my_kernel_bench
copy scripts\config.local.ps1.example scripts\config.local.ps1
# 编辑 OHOS_NATIVE、REMOTE_DIR
.\scripts\build.ps1
.\scripts\deploy.ps1
```

手机上：

```bash
cd /data/vendor/camera   # 或你在 config 里设的路径
./my_kernel_bench
```

## 框架约定

```
my_xxx_bench/
  lib/your_optimized/     # 验证通过后，整包拷进大项目
  src/
    baseline.*            # 大项目现状（REF）
    optimized.*           # 你的优化
    benchmark.cpp         # 注册 variant 表
    bench_case.*          # 输入数据（合成 or samples/）
  framework/              # 计时、PSNR、表格（一般不用改）
  samples/                # 真机输入
  scripts/                # build / deploy / pull
```

### Variant 四类

| kind | 含义 | 典型阈值 |
|------|------|----------|
| `REF` | baseline，只计时 | - |
| `SAME` | 必须像素一致 | maxDiff=0 |
| `FAST` | 更快，允许小偏差 | PSNR + maxDiff |
| `APRX` | 明显近似 | 放宽阈值 |

### 闭环流程

```
大项目慢路径 → 抽 lib + 最小输入 → bench 迭代 → PASS → lib 合回主工程
```

详见 [WORKFLOW.md](WORKFLOW.md)。

## 与 seamless_clone_bench 的关系

`seamless_clone_bench/` 是本框架的**第一个实例**（seamlessClone 优化）。新优化请用 `init_bench.ps1` 生成独立目录，不要和 seamless clone 混在一起。
