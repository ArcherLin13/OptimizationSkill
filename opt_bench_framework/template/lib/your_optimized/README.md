# your_optimized

Drop-in library validated by `__BENCH_NAME__`.

## Integrate into main project

1. Copy this folder to your repo
2. Add `.cpp` to your CMake target
3. Replace hot-path call with `your_optimized::Context`

```cpp
static your_optimized::Context g;
g.gaussianBlur5x5(input, output);
```

## Dependencies

OpenCV core + imgproc (adjust to match your module).
