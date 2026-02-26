import hnswlib
import numpy as np
import os

from ann_benchmarks.constants import SIM_SSD_DIR_CONTAINER
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
            max_elements=len(X), ef_construction=self.method_param["efConstruction"], M=self.method_param["M"],
            page_size=16384
        )
        data_labels = np.arange(len(X))
        self.p.add_items(np.asarray(X), data_labels)
        self.p.set_num_threads(1)

        # Save index to SIM_SSD_DIR and then reload it with cache
        index_path = os.path.join(SIM_SSD_DIR_CONTAINER, "hnswlib", "index")
        os.makedirs(os.path.join(SIM_SSD_DIR_CONTAINER, "hnswlib"), exist_ok=True)
        self.p.save_index(index_path)
        self.p.load_index(index_path, cache_size=125 * 1024 * 1024, thread_num=6)

    def set_query_arguments(self, ef):
        self.p.reset_metrics_counter()
        self.p.set_ef(ef)
        self.name = "hnswlib (%s, 'efQuery': %s)" % (self.method_param, ef)

    def query(self, v, n):
        # print(np.expand_dims(v,axis=0).shape)
        # print(self.p.knn_query(np.expand_dims(v,axis=0), k = n)[0])
        return self.p.knn_query(np.expand_dims(v, axis=0), k=n)[0][0]
    
    def batch_query(self, X, n: int) -> None:
        # X shape: (num_queries, dim)
        # knn_query returns: (labels_array, distances_array)
        # labels_array shape: (num_queries, n), dtype: int64
        self.res, _ = self.p.knn_query(X, k=n, num_threads=6)

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        """从C++ binding获取性能指标"""
        # 建议的C++ binding方法名称（需要在Python binding中实现）：
        # - get_cache_hit_rate(): 返回浮点数百分比
        # - get_io_op_num(): 返回整数I/O操作数
        # - get_memory_transfer_kb(): 返回浮点数KB单位

        latency_list = self.p.get_latency_ms()
        avg_latency_ms = sum(latency_list) / len(latency_list) if latency_list else 0
        
        detailed = self.p.get_detailed_latency()
        avg_cpu_ms = sum(d.cpu_ms for d in detailed) / len(detailed) if detailed else 0
        avg_io_ms = sum(d.io_ms for d in detailed) / len(detailed) if detailed else 0
        return {
            "cache_hit_rate": float(self.p.get_cache_hit_rate()),
            "io_operations": int(self.p.get_io_op_num()),
            "memory_transfer_kb": float(self.p.get_memory_transfer_kb()),
            "avg_latency_ms": float(avg_latency_ms),
            "avg_cpu_ms": float(avg_cpu_ms),
            "avg_io_ms": float(avg_io_ms),
            "cpu_qps": float(self.p.get_qps(avg_latency_ms, 6)),
            "avg_depth_mean": float(self.p.get_avg_depth_mean()),
            "avg_depth_std": float(self.p.get_avg_depth_std()),
        }

    def freeIndex(self):
        del self.p
