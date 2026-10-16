#!/usr/bin/env bash
# Download a prebuilt ONNX Runtime C/C++ package.
# Usage: fetch_ort.sh <dest> [version] [os-arch]
# os-arch examples: linux-x64 linux-aarch64 osx-arm64 osx-x86_64 win-x64 win-arm64
set -euo pipefail
DEST="${1:-third_party/onnxruntime}"
VERSION="${2:-1.22.0}"
TARGET="${3:-}"
if [[ -z "${TARGET}" ]]; then
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"
    case "${os}-${arch}" in
        linux-x86_64) TARGET="linux-x64" ;;
        linux-aarch64|linux-arm64) TARGET="linux-aarch64" ;;
        darwin-arm64) TARGET="osx-arm64" ;;
        darwin-x86_64) TARGET="osx-x86_64" ;;
        *) echo "set the target explicitly" >&2; exit 1 ;;
    esac
fi
mkdir -p "${DEST}"
case "${TARGET}" in
    win-*) EXT="zip" ;;
    *) EXT="tgz" ;;
esac
NAME="onnxruntime-${TARGET}-${VERSION}.${EXT}"
URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${NAME}"
TMP="$(mktemp -d)"
echo "fetch ${URL}"
curl -L --fail --retry 3 -o "${TMP}/${NAME}" "${URL}"
if [[ "${EXT}" == "zip" ]]; then
    unzip -q "${TMP}/${NAME}" -d "${TMP}"
else
    tar -xzf "${TMP}/${NAME}" -C "${TMP}"
fi
SRC="$(find "${TMP}" -maxdepth 1 -type d -name 'onnxruntime-*' | head -n 1)"
rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"
mv "${SRC}" "${DEST}"
echo "ORT ready at ${DEST}"
