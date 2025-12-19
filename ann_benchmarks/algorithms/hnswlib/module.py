import hnswlib
import numpy as np

from ..base.module import BaseANN


class HnswLib(BaseANN):
    def __init__(self, metric, method_param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.method_param = method_param
        # print(self.method_param,save_index,query_param)

    def fit(self, X):
        # Only l2 is supported currently
        self.p = hnswlib.Index(space=self.metric, dim=len(X[0]))
        self.p.init_index(
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"]
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

    def set_query_arguments(self, ef):
        self.p.set_ef(ef)
        self.name = "hnswlib (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]

    def get_additional(self):
        """从C++ binding获取性能指标"""
        # 建议的C++ binding方法名称（需要在Python binding中实现）：
        # - get_cache_hit_rate(): 返回浮点数百分比
        # - get_io_operations(): 返回整数I/O操作数
        # - get_memory_transfer_kb(): 返回浮点数KB单位

        return {
            "cache_hit_rate": float(self.p.get_cache_hit_rate()),
            "io_operations": int(self.p.get_io_operations()),
            "memory_transfer_kb": float(self.p.get_memory_transfer_kb())
        }

    def done(self):
        """清理资源，C++ binding需要调用清理方法"""
        # 建议在C++ binding中实现：reset_counters()
        if hasattr(self.p, 'reset_counters'):
            self.p.reset_counters()
        super().done()

    def freeIndex(self):
        del self.p
