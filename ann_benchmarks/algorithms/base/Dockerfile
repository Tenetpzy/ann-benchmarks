FROM ubuntu:24.04

RUN apt-get update
# 注意：这里需要安装 python3-venv 或者 python3-full 才能创建虚拟环境
RUN apt-get install -y python3-pip python-is-python3 build-essential git python3-venv

# 1. 创建虚拟环境 (建议放在 /opt/venv)
RUN python3 -m venv /opt/venv

# 2. 关键步骤：设置环境变量
# 将虚拟环境的 bin 目录加到 PATH 的最前面
# 这相当于在接下来的所有层级中 "永久激活" 了虚拟环境
ENV PATH="/opt/venv/bin:$PATH"

# 验证一下：此时运行 python 应该是指向虚拟环境的 python
RUN python --version | grep 'Python 3.12'

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./

# 3. 安装依赖
# 由于 PATH 已经设置，这里的 pip 会自动安装到 /opt/venv 中
# 不需要加 --break-system-packages，也不会报错
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 4. 启动
# 这里的 "python" 会自动解析为 /opt/venv/bin/python
ENTRYPOINT ["python", "-u", "run_algorithm.py"]