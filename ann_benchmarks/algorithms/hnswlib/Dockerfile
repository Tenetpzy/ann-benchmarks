FROM ann-benchmarks

# 只安装基础依赖，删除构建时的clone和安装
RUN pip install pybind11 numpy setuptools

# 添加启动脚本，在容器启动时执行安装逻辑
COPY ann_benchmarks/algorithms/hnswlib/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 覆盖ENTRYPOINT，在启动时先执行安装，再运行算法
ENTRYPOINT ["/entrypoint.sh"]

