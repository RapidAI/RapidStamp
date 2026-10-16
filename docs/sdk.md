# C/C++ SDK 与 Python 包

推理用两个 ONNX：`seal_det.onnx`（印章实例检测）和 `seal_rec.onnx`（展开后的文字识别）。几何后处理（letterbox、极坐标展开、方章切格、NMS、CTC 贪心解码）在 SDK 里完成，不放进计算图，方便各平台共用同一对模型。

识别器按 ONNX 输入自己分支，Python 和 C++ 同一套规则。1 通道是灰度 CTC，条带缩到 32×384，输入名一般是 `image`。3 通道是 PP-OCRv5 那种文档识别：先缩到高 48、宽按比例且不超过 3200，再转 BGR，做 `(x/255-0.5)/0.5`。输出取最后一维等于字表长度加 1 的那个张量，空白仍是 index 0。`RapidStamp(..., fast=True)` 改读同目录的 `seal_rec_fast.onnx` 和 `charset_fast.txt`，这是轻量 CTC 的快速模式。C API 不增加参数，调用方把要跑的识别模型和字表路径传进来即可。

运行时数值合同在 `preprocess.txt`。Python 与 C++ 都读这个文件。默认值和 `cpp/src/constants.h`、`python/src/rapidstamp/labels.py` 一致。

## Python

```bash
pip install -e python
rapidstamp path/to/page.jpg --model-dir models
```

环境变量 `RAPIDSTAMP_MODEL_DIR` 可以代替 `--model-dir`。没有 GPU 版本的依赖，默认 `CPUExecutionProvider`。

## C API

头文件只有 `cpp/include/rapidstamp.h`。

```c
rapidstamp_engine* engine = rapidstamp_create(
    "models/seal_det.onnx", "models/seal_rec.onnx", "models/charset.txt", NULL);
rapidstamp_seal seals[16];
int n = rapidstamp_infer(engine, rgb, width, height, width * 3, seals, 16);
rapidstamp_destroy(engine);
```

`rgb` 是 8-bit RGB，行优先。返回值是检出的印章个数；负数表示参数或运行时错误，`rapidstamp_last_error()` 给出原因。单个 engine 内部有锁，可以跨线程调用；不要在 `destroy` 之后再使用它。

`rapidstamp_config` 里某项为 0 时，使用 `preprocess.txt` 的值。`angle_steps` 是圆形/椭圆章极角搜索的步数，训练分布对齐在 `start_deg`（默认 140°）附近，搜索用来覆盖旋转。

## 桌面端编译

ONNX Runtime 使用 1.22.0 的预编译 C/C++ 包。这一版同时提供 Linux x64/arm64、Windows x64/arm64、macOS arm64 和 macOS x86_64。更新的 1.31 包不再附带 macOS x86_64，所以 SDK 固定在 1.22。

```bash
bash cpp/scripts/fetch_ort.sh /tmp/ort 1.22.0          # 按本机选择包
# 也可以显式指定：linux-x64 linux-aarch64 osx-arm64 osx-x86_64 win-x64 win-arm64
cmake -S cpp -B cpp/build -DORT_ROOT=/tmp/ort -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --parallel
cpp/build/rapidstamp_test_geom cpp/testdata/unwrap_golden.txt
cpp/build/rapidstamp_demo models/seal_det.onnx models/seal_rec.onnx models/charset.txt page.ppm
```

产物是 `librapidstamp.so` / `rapidstamp.dll` / `librapidstamp.dylib`，以及同名静态库。安装规则把库放到 `lib`，把头文件放到 `include`。

## Android

`cpp/scripts/build_android.sh` 用 Maven 上的 `onnxruntime-android` AAR（arm64-v8a 的 `libonnxruntime.so`）和桌面包装里的 C++ 头文件，配合 NDK r26 交叉编译。

```bash
export ANDROID_NDK_HOME=/opt/android-ndk-r26d
bash cpp/scripts/build_android.sh
```

CI 的 `android-arm64` 任务做同样的事。应用把 `librapidstamp.so` 和 `libonnxruntime.so` 一起打包进 `jniLibs/arm64-v8a`，再用 JNI 转调 `rapidstamp.h`。输入仍然是 RGB 字节，不在 SDK 里解码 JPEG。

## iOS

仓库不在 CI 里编 iOS：官方分发方式是 CocoaPod `onnxruntime-c`（或带 Core ML / XNNPACK 的 xcframework），需要本机 Xcode。桌面端的 C API 和几何代码可以直接编进 App。拿到 iOS 版 ONNX Runtime 的 include 与库之后：

```bash
cmake -S cpp -B build-ios \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DORT_ROOT=/path/to/onnxruntime-ios \
  -DRAPIDSTAMP_BUILD_TESTS=OFF \
  -DRAPIDSTAMP_BUILD_EXAMPLES=OFF
```

`ORT_ROOT` 需要和桌面包一样有 `include/` 与 `lib/`。若只有 xcframework，把头文件指到 pod 的 Headers，把链接库指到 `onnxruntime.framework`。

## Python 轮子

`python/` 是纯 Python 包，推理依赖 PyPI 上的 `onnxruntime`。GitHub Actions 在 Linux、Windows、macOS 上用 `python -m build` 打 `py3-none-any` 轮子并安装。模型文件放在 `python/src/rapidstamp/models/` 时会打进轮子；否则运行时从 `RAPIDSTAMP_MODEL_DIR` 或仓库的 `models/` 读取。
