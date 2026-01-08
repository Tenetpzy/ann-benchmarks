import numpy as np

def io_num(effective_recall, C, K, B):
    error_rate = 1.0 - effective_recall
    return C + K * np.power(error_rate, -B)

def simulate_performance(
    # 召回率
    recall_target: float,

    # 资源限制
    available_cache_pages: int,
    cpu_cores: int,

    # 算法模型参数
    beam_width: int,            # 搜索束宽，表示一次向SSD并行提交的IO个数
    cache_hit_rate: float,      # 缓存命中率 (0.0 - 1.0)
    C: float = 1,  # 算法IO复杂度系数，固有I/O
    B: float = 1,  # 算法IO复杂度系数，维度相关
    K: float = 1,  # 数据集稀疏度，值越大表示搜索的IO次数越多
    T_cpu_base_us: float = 500.0, # 在内存图中的导航时间
    T_cpu_per_node_us: float = 20.0, # 每处理一个节点的CPU时间开销
    page_size_per_io: int = 16 * 1024,  # 每次IO读取的页面大小(字节) 

    # SSD算力限制
    ssd_cpu_cores: int = 0,
    csd_schedular_on: bool = True,

    # 硬件物理参数
    T_ssd_latency_base_us: float = 200,
    ssd_iops_base: float = 100000,

    # 随机性参数
    sim_rounds: int = 1000,
    jitter_io_complexity: float = 0.2,
    jitter_latency_sigma: float = 0.4,
    jitter_system_noise: float = 0.05,
    jitter_cpu_overhead: float = 0.1,      # T_cpu_base_us 的随机波动
    jitter_cpu_per_node: float = 0.1,      # T_cpu_per_node_us 的随机波动
    jitter_cache_hit: float = 0.05         # cache_hit_rate 的随机波动
) -> dict:
    
    # 限制召回率上限，防止 Log(0)
    effective_recall = min(recall_target, 0.99995)
    real_pages_per_req = beam_width

    # 向量化生成随机 IO
    error_rate = 1.0 - effective_recall
    base_logical_ios = C + K * np.power(error_rate, -B)

    # 带有波动的逻辑IO
    sim_logical_ios = np.random.normal(loc=base_logical_ios, scale=base_logical_ios * jitter_io_complexity, size=sim_rounds)
    sim_logical_ios = np.maximum(sim_logical_ios, 1.0)

    # 即使物理IO为0，CPU依然需要处理所有逻辑节点
    sim_cpu_nodes_processed = sim_logical_ios

    # 硬件波动
    # 确保最小值 > 0，避免 log(0) = -inf
    min_ssd_latency = max(T_ssd_latency_base_us, 1e-6)
    mu = np.log(min_ssd_latency)
    sim_ssd_latencies_us = np.random.lognormal(mean=mu, sigma=jitter_latency_sigma, size=sim_rounds)
    
    sim_ssd_iops_cap = np.random.normal(loc=ssd_iops_base, scale=ssd_iops_base * jitter_system_noise, size=sim_rounds)
    sim_cpu_cores_cap = np.random.normal(loc=cpu_cores, scale=cpu_cores * jitter_system_noise, size=sim_rounds)
    
    # 保护下限
    sim_ssd_iops_cap = np.maximum(sim_ssd_iops_cap, 1000.0)
    sim_cpu_cores_cap = np.maximum(sim_cpu_cores_cap, 1.0)
    sim_ssd_cpu_cores_cap = np.random.normal(loc=ssd_cpu_cores, scale=ssd_cpu_cores * jitter_system_noise, size=sim_rounds)
    sim_ssd_cpu_cores_cap = np.maximum(sim_ssd_cpu_cores_cap, 1.0)

    # CPU开销随机波动 (对数正态分布，保持正值)
    # 确保最小值 > 0，避免 log(0) = -inf
    min_cpu_base = max(T_cpu_base_us, 1e-6)
    min_cpu_node = max(T_cpu_per_node_us, 1e-6)

    mu_cpu_base = np.log(min_cpu_base)
    sim_cpu_base_us = np.random.lognormal(mean=mu_cpu_base, sigma=jitter_cpu_overhead, size=sim_rounds)

    mu_cpu_node = np.log(min_cpu_node)
    sim_cpu_per_node_us = np.random.lognormal(mean=mu_cpu_node, sigma=jitter_cpu_per_node, size=sim_rounds)

    # 缓存命中率随机波动 (有界高斯分布，限制在[0,1])
    sim_cache_hit_rate = np.random.normal(loc=cache_hit_rate, scale=cache_hit_rate * jitter_cache_hit, size=sim_rounds)
    sim_cache_hit_rate = np.clip(sim_cache_hit_rate, 0.0, 0.9999)

    # 应用缓存命中率计算物理 IO
    sim_physical_ios = sim_logical_ios * (1.0 - sim_cache_hit_rate)
    sim_physical_ios = np.maximum(sim_physical_ios, 0.0)

    # Latency
    sim_io_rounds = np.ceil(sim_physical_ios / beam_width)
    sim_latencies_us = (
        sim_cpu_base_us +
        (sim_io_rounds * sim_ssd_latencies_us) +
        (sim_cpu_nodes_processed * sim_cpu_per_node_us)
    )
    sim_latencies_sec = sim_latencies_us / 1_000_000.0

    memory_transferred_bytes = int(np.mean(sim_physical_ios * page_size_per_io))

    # QPS (Bottleneck Analysis)
    # Disk Bound
    with np.errstate(divide='ignore'):
        qps_disk = np.where(sim_physical_ios > 0.1,
                            sim_ssd_iops_cap / sim_physical_ios,
                            99999999.0)

    # CPU Bound (使用平均CPU开销)
    avg_cpu_base = np.mean(sim_cpu_base_us)
    avg_cpu_per_node = np.mean(sim_cpu_per_node_us)

    if csd_schedular_on:
        # 调度器开启：CPU计算均匀分布，总资源 = CPU核心 + SSD CPU核心
        cpu_base_time_us = avg_cpu_base
        cpu_node_time_us = sim_cpu_nodes_processed * avg_cpu_per_node
        total_cpu_cores = sim_cpu_cores_cap + sim_ssd_cpu_cores_cap
        cpu_active_time_us = cpu_base_time_us + cpu_node_time_us
        qps_cpu = (total_cpu_cores * 1_000_000.0) / cpu_active_time_us
        qps_ssd_computing = None
        memory_transferred_rate_per_req = np.mean((0.1 * memory_transferred_bytes + memory_transferred_bytes * cpu_cores / float(cpu_cores + ssd_cpu_cores)) / sim_latencies_sec)
    else:
        # 调度器关闭：base 在主CPU，node processing 在SSD CPU
        cpu_base_time_us = avg_cpu_base
        cpu_node_time_us = sim_cpu_nodes_processed * avg_cpu_per_node

        # 主CPU QPS
        qps_cpu = (sim_cpu_cores_cap * 1_000_000.0) / (cpu_base_time_us + cpu_node_time_us)
        # SSD CPU QPS (只算 node processing 部分)
        qps_ssd_computing = (sim_ssd_cpu_cores_cap * 1_000_000.0) / cpu_node_time_us

        memory_transferred_rate_per_req = np.mean(0.1 * memory_transferred_bytes / sim_latencies_sec)

    # 内存带宽随机波动（对数正态分布，保持正值）
    jitter_memory_bandwidth = 0.05
    min_mem_rate = max(memory_transferred_rate_per_req, 1e-6)
    mu_mem = np.log(min_mem_rate)
    memory_transferred_rate_per_req = np.random.lognormal(mean=mu_mem, sigma=jitter_memory_bandwidth, size=1)[0]

    # Memory/Concurrency Bound
    max_concurrency = max(1, int(available_cache_pages // real_pages_per_req))
    qps_buffer = max_concurrency / sim_latencies_sec

    # 构建 bottleneck matrix
    if qps_ssd_computing is not None:
        # 4个瓶颈：Disk, CPU, SSD_Computing, Buffer
        bottleneck_matrix = np.vstack([qps_disk, qps_cpu, qps_ssd_computing, qps_buffer])
        bottleneck_indices = np.argmin(bottleneck_matrix, axis=0)
        b_counts = np.bincount(bottleneck_indices, minlength=4)
        total_samples = float(sim_rounds)
        analysis = {
            "Disk_Bound_Prob": round(b_counts[0] / total_samples, 3),
            "CPU_Bound_Prob": round(b_counts[1] / total_samples, 3),
            "SSD_Computing_Prob": round(b_counts[2] / total_samples, 3),
            "Buffer_Bound_Prob": round(b_counts[3] / total_samples, 3)
        }
        final_qps = np.minimum.reduce([qps_disk, qps_cpu, qps_ssd_computing, qps_buffer])
    else:
        # 3个瓶颈：Disk, CPU, Buffer
        bottleneck_matrix = np.vstack([qps_disk, qps_cpu, qps_buffer])
        bottleneck_indices = np.argmin(bottleneck_matrix, axis=0)
        b_counts = np.bincount(bottleneck_indices, minlength=3)
        total_samples = float(sim_rounds)
        analysis = {
            "Disk_Bound_Prob": round(b_counts[0] / total_samples, 3),
            "CPU_Bound_Prob": round(b_counts[1] / total_samples, 3),
            "Buffer_Bound_Prob": round(b_counts[2] / total_samples, 3)
        }
        final_qps = np.minimum(qps_disk, np.minimum(qps_cpu, qps_buffer))

    return {
        "input_summary_recall_target": recall_target,
        "input_summary_avg_physical_io": round(float(np.mean(sim_physical_ios)), 1),
        "input_summary_max_concurrency": max_concurrency,
        "input_summary_cache_hit_rate": cache_hit_rate,
        "latency_ms_mean": round(float(np.mean(sim_latencies_us) / 1000), 2),
        "latency_ms_p50": round(float(np.percentile(sim_latencies_us, 50) / 1000), 2),
        "latency_ms_p99": round(float(np.percentile(sim_latencies_us, 99) / 1000), 2),
        "latency_ms_max": round(float(np.max(sim_latencies_us) / 1000), 2),
        "qps_mean": int(np.mean(final_qps)),
        "qps_p50": int(np.percentile(final_qps, 50)),
        "qps_p99": int(np.percentile(final_qps, 99)),
        "qps_stddev": int(np.std(final_qps)),
        "memory_bandwidth": float(memory_transferred_rate_per_req),  # B/s
        "bottleneck": analysis
    }