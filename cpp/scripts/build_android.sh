#!/usr/bin/env bash
# Cross-compile librapidstamp for Android arm64 against the ONNX Runtime AAR.
# Requires ANDROID_NDK_HOME (or ANDROID_NDK) and curl/unzip.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="${ORT_VERSION:-1.22.0}"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
if [[ -z "${NDK}" ]]; then
    echo "Set ANDROID_NDK_HOME" >&2
    exit 1
fi
WORK="${ROOT}/build-android"
mkdir -p "${WORK}"
bash "${ROOT}/scripts/fetch_ort.sh" "${WORK}/ort-headers" "${VERSION}" linux-x64
AAR="${WORK}/onnxruntime-android.aar"
curl -L --fail --retry 3 -o "${AAR}" \
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${VERSION}/onnxruntime-android-${VERSION}.aar"
unzip -qo "${AAR}" -d "${WORK}/aar"
STAGE="${WORK}/ort-android"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/include" "${STAGE}/lib"
cp -a "${WORK}/ort-headers/include/." "${STAGE}/include/"
cp "${WORK}/aar/jni/arm64-v8a/libonnxruntime.so" "${STAGE}/lib/"
cmake -S "${ROOT}" -B "${WORK}/cmake" \
    -DCMAKE_TOOLCHAIN_FILE="${NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DORT_ROOT="${STAGE}" \
    -DRAPIDSTAMP_BUILD_TESTS=OFF \
    -DRAPIDSTAMP_BUILD_EXAMPLES=OFF
cmake --build "${WORK}/cmake" --parallel
echo "built ${WORK}/cmake/librapidstamp.so"
