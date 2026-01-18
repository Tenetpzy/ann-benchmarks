FROM ann-benchmarks

COPY ann_benchmarks/algorithms/hnswlib/entrypoint.sh /entrypoint.sh
RUN pip install pybind11 numpy setuptools && \
    chmod +x /entrypoint.sh && \
    git clone https://github.com/axboe/liburing.git && \
    cd liburing && \
    make && \
    make install && \
    cd .. && \
    rm -rf liburing

# 覆盖ENTRYPOINT，在启动时先执行安装，再运行算法
ENTRYPOINT ["/entrypoint.sh"]

