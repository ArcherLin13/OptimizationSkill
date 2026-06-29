# OpenCV for this project

## 已包含（仓库内）

`include/opencv4/` — OpenCV **4.9** 编译所需头文件（约 150 个文件，**不是**完整 OpenCV 源码仓库）。

覆盖模块：**core、imgproc、photo**（与 benchmark 源码一致）。

## 你需要自己放入（来自手机 `/chip_prod/lib64`）

把这 4 个 so 拷到本目录（或 `lib/` 子目录）：

```text
libopencv_core.so*
libopencv_imgcodecs.so*
libopencv_imgproc.so*
libopencv_photo.so*
```

## 编译

```powershell
.\scripts\build.ps1
.\scripts\deploy_chip_prod.ps1
```

**注意：** 头文件是 4.9。若手机 so 是别的版本，请换成同版本头文件，否则可能编译/运行 ABI 不一致。
