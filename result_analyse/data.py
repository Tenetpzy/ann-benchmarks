recall_target = [0.9, 0.92, 0.95, 0.97, 0.99, 0.992, 0.995, 0.997, 0.999]
beam_width = 8
available_cache_pages = 128
cpu_cores = 6

# available_cache_pages
# cpu_cores
# cache_hit_rate
# C: float = 1, 算法IO复杂度系数
# K: float = 1, 数据集稀疏度，值越大表示搜索的IO次数越多
# T_cpu_base_us: float = 500.0  // 在内存图中的导航时间

hnsw_baseline_C = 10
hnsw_reorder_C = 6

hnsw_baseline_cache_hit_rate = [0.45, 0.38, 0.36, 0.35, 0.33, 0.32, 0.30, 0.29, 0.27]
hnsw_reorder_cache_hit_rate = [0.50, 0.42, 0.41, 0.40, 0.38, 0.38, 0.36, 0.35, 0.34]

hnsw_baseline_cpu_base_us = 0
hnsw_reorder_cpu_base_us = 0
