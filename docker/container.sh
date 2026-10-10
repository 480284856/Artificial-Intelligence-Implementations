#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

docker run -idt \
    --shm-size=512m \
    -p 127.0.0.1:6901:6901 \
    -e VNC_PW=password \
    -e VNCOPTIONS=-disableBasicAuth \
    --name aif \
    -u root \
    -v "${PROJECT_DIR}:/workspace" \
    -w /workspace \
    kasmweb/ubuntu-jammy-desktop:aarch64-1.18.0
