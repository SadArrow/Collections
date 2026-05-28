#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-/root/workspace/qianyupeng}"
TARGET_ROOT="${2:-${BASE}/downloads/x11_runtime_libs}"
WORK="${BASE}/tmp/deb_runtime_libs"
TARGET="${TARGET_ROOT}/usr/lib/x86_64-linux-gnu"

rm -rf "${WORK}"
mkdir -p "${WORK}" "${TARGET}"
cd "${WORK}"

apt-get download libsm6 libice6 libglu1-mesa

mkdir -p extract
dpkg-deb -x libsm6_*.deb extract
dpkg-deb -x libice6_*.deb extract
dpkg-deb -x libglu1-mesa_*.deb extract

cp -a extract/usr/lib/x86_64-linux-gnu/libSM.so* "${TARGET}/"
cp -a extract/usr/lib/x86_64-linux-gnu/libICE.so* "${TARGET}/"
cp -a extract/usr/lib/x86_64-linux-gnu/libGLU.so* "${TARGET}/"

echo "LIBS_READY"
ls -l "${TARGET}"/libSM.so* "${TARGET}"/libICE.so* "${TARGET}"/libGLU.so*
echo "TARGET=${TARGET}"
echo "GLIBC_REQS"
strings -a "${TARGET}/libICE.so.6" | grep GLIBC_ | sort -u | tail -10 || true
