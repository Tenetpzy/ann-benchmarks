recall_target = [0.9, 0.92, 0.95, 0.97, 0.99, 0.992, 0.995, 0.997, 0.999]
available_cache_pages = 128

# available_cache_pages
# beam_width
# cpu_cores
# cache_hit_rate
# C: float = 1, 算法IO复杂度系数
# K: float = 1, 数据集稀疏度，值越大表示搜索的IO次数越多
# T_cpu_base_us: float = 500.0  // 在内存图中的导航时间
# T_cpu_per_node_us: float = 20.0  // 每处理一个节点的CPU时间开销

csdann_opt_storage_beam_width = 8
csdann_opt_storage_cpu_cores = 7
csdann_opt_storage_cache_hit_rate = [0.60, 0.58, 0.55, 0.53, 0.50, 0.45, 0.40, 0.37, 0.32]
csdann_opt_storage_C = 7.8
csdann_opt_storage_cpu_base_us = 800
csdann_opt_storage_cpu_per_node_us = 4

hnsw_baseline_C = 10
csdann_reorder_C = 8

hnsw_baseline_cpu_cores = 6
csdann_reorder_cpu_cores = 6

hnsw_baseline_beam_width = 1
csdann_reorder_beam_width = 8

hnsw_baseline_cache_hit_rate = [0.45, 0.38, 0.36, 0.35, 0.33, 0.30, 0.25, 0.20, 0.17]
csdann_reorder_cache_hit_rate = [0.50, 0.48, 0.45, 0.43, 0.40, 0.35, 0.30, 0.27, 0.22]

hnsw_baseline_cpu_base_us = 200
csdann_reorder_cpu_base_us = 200

hnsw_baseline_cpu_per_node_us = 20
csdann_reorder_cpu_per_node_us = 20
