#!/bin/bash
set -e

echo "=== HNSWLIB ENTRYPOINT START ===" >&2

# 检查挂载的本地hnswlib仓库是否存在
if [ -d "/hnswlib" ]; then
    echo "Found local hnswlib repository at /hnswlib" >&2
    echo "Building and installing hnswlib..." >&2
    cd /hnswlib/python_bindings
    python3 setup.py install
    echo "Installed hnswlib from local repository" >&2
else
    echo "ERROR: No local hnswlib repository found at /hnswlib" >&2
    echo "Please ensure your local hnswlib repository is mounted to /hnswlib" >&2
    exit 1
fi

# 验证安装
echo "Verifying hnswlib installation..." >&2
python3 -c 'import hnswlib; print(f"Successfully imported hnswlib version: {hnswlib.__version__}")' || { echo "Failed to import hnswlib"; exit 1; }

echo "hnswlib installed successfully, starting benchmark..." >&2
echo "=== HNSWLIB ENTRYPOINT END ===" >&2

# 执行原来的命令（通过base镜像的ENTRYPOINT）
exec python -u /home/app/run_algorithm.py "$@"
