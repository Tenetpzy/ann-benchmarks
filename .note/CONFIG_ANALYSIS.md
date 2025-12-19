# ANN-Benchmarks 配置文件格式与运行流程分析

## 一、配置文件格式详解

### 1.1 基本结构

配置文件采用多层嵌套的YAML格式，结构如下：

```yaml
{point_type}:          # 数据类型层：float, bit, uint8等
  {distance_metric}:   # 距离度量层：euclidean, angular, hamming, jaccard, 或 "any"
    - name: 算法显示名称
      constructor: Python类名
      module: Python模块路径
      docker_tag: Docker镜像标签
      base_args: [基础参数列表]
      disabled: true/false
      run_groups:      # 运行组配置
        组名1:
          args/arg_groups: [构建参数]
          query_args/query_arg_groups: [查询参数]
        组名2:
          ...
```

### 1.2 配置格式变体分析

#### 格式1: 使用 `args` + `query_args` (列表形式)

**示例：FAISS**
```yaml
float:
  any:
  - base_args: ['@metric']
    constructor: FaissIVF
    module: ann_benchmarks.algorithms.faiss
    name: faiss-ivf
    run_groups:
      base:
        args: [[32, 64, 128, 256, 512]]      # 第一个参数的可能值
        query_args: [[1, 5, 10, 50, 100]]    # 查询参数的可能值
```

**解析方式**：
- `args` 是一个列表的列表，每个子列表代表一个位置参数的所有可能取值
- 通过笛卡尔积生成所有参数组合
- 例如 `args: [[A, B], [X, Y]]` 会生成: `[A, X]`, `[A, Y]`, `[B, X]`, `[B, Y]`

#### 格式2: 使用 `arg_groups` (字典形式)

**示例：PGVector**
```yaml
float:
  any:
  - base_args: ['@metric']
    constructor: PGVector
    module: ann_benchmarks.algorithms.pgvector
    name: pgvector
    run_groups:
      M-16:
        arg_groups: [{M: 16, efConstruction: 200}]
        query_args: [[10, 20, 40, 80, 120]]
```

**解析方式**：
- `arg_groups` 包含字典，字典的值可以是列表
- 会展开成多个参数组合
- 例如 `{M: [16, 32], ef: [100, 200]}` 会生成4个组合

#### 格式3: 混合字典和列表

**示例：NMSLib**
```yaml
float:
  euclidean:
  - base_args: ['@metric', hnsw]
    constructor: NmslibReuseIndex
    module: ann_benchmarks.algorithms.nmslib
    name: hnsw(nmslib)
    run_groups:
      M-12:
        arg_groups: [{M: 12, efConstruction: 800, post: 0}, false]
        query_args: [[1, 2, 5, 10, 15, 20]]
```

**解析方式**：
- `arg_groups` 可以包含字典和其他值的混合
- 字典会展开，其他值直接使用
- 最终传递给构造函数: `NmslibReuseIndex('euclidean', 'hnsw', {M: 12, ...}, false)`

#### 格式4: 多查询参数

**示例：FAISS IVFPQFS**
```yaml
- constructor: FaissIVFPQfs
  run_groups:
    base:
      args: [[512, 1024, 2048]]
      query_args: [[1, 5, 10], [0, 10, 100]]  # 两个查询参数
```

**解析方式**：
- 多个 `query_args` 列表会生成笛卡尔积
- `[[1, 5], [0, 10]]` 会生成: `[1, 0]`, `[1, 10]`, `[5, 0]`, `[5, 10]`

### 1.3 特殊变量替换

配置中可以使用占位符变量，运行时会被替换：

- `@metric`: 当前距离度量（euclidean/angular等）
- `@dimension`: 数据集维度
- `@count`: k值（返回的最近邻数量）

**示例**：
```yaml
base_args: ['@metric', '@dimension']
# 运行时替换为: ['euclidean', 128]
```

### 1.4 距离度量匹配规则

1. **精确匹配**：优先使用与数据集完全匹配的度量
   ```yaml
   float:
     euclidean:  # 仅用于euclidean数据集
   ```

2. **通配符匹配**：使用 `any` 可匹配所有度量
   ```yaml
   float:
     any:  # 用于所有float类型的数据集
   ```

3. **同时定义**：可以同时定义 `any` 和特定度量，允许不同配置
   ```yaml
   float:
     any:
       - name: algo-general
         run_groups: {...}
     euclidean:
       - name: algo-euclidean-optimized
         run_groups: {...}  # euclidean数据集会同时运行这两个配置
   ```

---

## 二、Benchmark运行流程详解

### 2.1 总体流程图

```
用户命令: python run.py --dataset glove-100-angular
    ↓
main.py::main()
    ├─ 1. 解析命令行参数
    ├─ 2. 加载数据集元信息 (get_dataset)
    │     → 获取: point_type=float, distance=angular, dimension=100
    ├─ 3. 生成算法定义 (get_definitions)
    │     ├─ 扫描所有 config.yml 文件
    │     ├─ 过滤匹配的 point_type 和 distance
    │     ├─ 展开所有参数组合 (笛卡尔积)
    │     └─ 生成 Definition 对象列表
    ├─ 4. 过滤定义
    │     ├─ 过滤已运行的 (除非 --force)
    │     ├─ 过滤禁用的 (除非 --run-disabled)
    │     ├─ 过滤缺失Docker镜像的
    │     └─ 随机打乱顺序
    ├─ 5. 创建工作队列和进程池
    │     └─ 启动 N 个 worker 进程 (--parallelism N)
    └─ 6. 每个 worker 执行
          └─ run_docker() 或 run()
                ├─ 启动Docker容器 (或本地执行)
                ├─ runner.py::run()
                │     ├─ 加载并转换数据集
                │     ├─ 实例化算法
                │     ├─ 构建索引 (fit)
                │     ├─ 运行查询 (query)
                │     └─ 存储结果
                └─ 容器清理
```

### 2.2 关键步骤详解

#### 步骤1: 配置解析 (definitions.py)

**函数调用链**：
```python
get_definitions(dimension, point_type, distance_metric, count, base_dir)
  ↓
_get_algorithm_definitions(point_type, distance_metric, base_dir)
  ├─ load_configs(point_type, base_dir)  # 加载所有config.yml
  └─ 返回匹配的算法定义字典
  ↓
create_definitions_from_algorithm(name, algo, dimension, distance_metric, count)
  ├─ 遍历 run_groups
  ├─ prepare_args(run_group) → 生成构建参数组合
  ├─ prepare_query_args(run_group) → 生成查询参数组合
  └─ _substitute_variables(args, {
        '@count': 10,
        '@metric': 'angular',
        '@dimension': 100
     })
```

**参数组合生成核心逻辑** (`_generate_combinations`):

```python
# 输入: args = [[32, 64], [100, 200]]
# 输出: [[32, 100], [32, 200], [64, 100], [64, 200]]

# 输入: args = {M: [16, 32], ef: [100, 200]}
# 输出: [{M: 16, ef: 100}, {M: 16, ef: 200}, {M: 32, ef: 100}, {M: 32, ef: 200}]
```

#### 步骤2: 过滤机制 (main.py)

```python
# 1. 过滤已运行的结果
filter_already_run_definitions()
  → 检查 results/{dataset}/10/{algorithm}_{params}.json 是否存在

# 2. 过滤Docker镜像
filter_by_available_docker_images()
  → 使用 docker.from_env() 检查镜像是否已构建

# 3. 过滤禁用的算法
filter_disabled_algorithms()
  → 检查 config.yml 中的 disabled: true
```

#### 步骤3: 并行执行 (main.py)

```python
# 创建工作队列
task_queue = multiprocessing.Queue()
for definition in definitions:
    task_queue.put(definition)

# 启动worker进程
workers = [
    multiprocessing.Process(
        target=run_worker,
        args=(cpu_id, mem_limit, args, task_queue)
    )
    for cpu_id in range(parallelism)
]

# run_worker 循环取任务
while not queue.empty():
    definition = queue.get()
    run_docker(definition, ...)  # 在Docker中运行
```

#### 步骤4: Docker容器执行 (runner.py)

```python
def run_docker(definition, dataset, count, runs, timeout, batch, cpu_limit, mem_limit):
    # 构建命令
    cmd = [
        '--dataset', dataset,
        '--algorithm', definition.algorithm,
        '--module', definition.module,
        '--constructor', definition.constructor,
        '--runs', str(runs),
        '--count', str(count),
        json.dumps(definition.arguments),
        *[json.dumps(qag) for qag in definition.query_argument_groups]
    ]

    # 启动容器
    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            'ann_benchmarks': {'bind': '/home/app/ann_benchmarks', 'mode': 'ro'},
            'data': {'bind': '/home/app/data', 'mode': 'ro'},
            'results': {'bind': '/home/app/results', 'mode': 'rw'},
            'sim_ssd': {'bind': '/home/app/sim_ssd', 'mode': 'rw'},
        },
        cpuset_cpus=cpu_limit,  # 例如: "0" 或 "0-7"
        mem_limit=mem_limit,
        detach=True
    )

    # 等待完成
    container.wait(timeout=timeout)
```

#### 步骤5: 索引构建和查询 (runner.py)

```python
def run(definition, dataset_name, count, run_count, batch):
    # 1. 实例化算法
    algo = instantiate_algorithm(definition)

    # 2. 加载数据
    X_train, X_test, distance = load_and_transform_dataset(dataset_name)

    # 3. 构建索引
    memory_before = algo.get_memory_usage()
    t0 = time.time()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_before

    # 4. 遍历所有查询参数组合
    for query_arguments in definition.query_argument_groups:
        if query_arguments:
            algo.set_query_arguments(*query_arguments)

        # 5. 运行查询 (多次取最佳)
        for run_idx in range(run_count):
            results = []
            for query_vector in X_test:
                t0 = time.time()
                candidates = algo.query(query_vector, count)
                query_time = time.time() - t0

                # 验证正确性：重新计算距离
                verified_candidates = [
                    (idx, distance_metric(query_vector, X_train[idx]))
                    for idx in candidates
                ]
                results.append((query_time, verified_candidates))

            best_search_time = min(best_search_time,
                                   sum(t for t, _ in results) / len(results))

        # 6. 存储结果
        store_results(dataset_name, count, definition, query_arguments,
                      descriptor, results, batch)
```

#### 步骤6: 结果存储格式

```json
{
  "algo": "pgvector",
  "dataset": "glove-100-angular",
  "count": 10,
  "batch_mode": false,
  "build_time": 125.3,
  "index_size": 524288000,
  "best_search_time": 0.0023,
  "candidates": 10.0,
  "name": "PGVector(m=16, ef_construction=200, ef_search=40)",
  "distance": "angular",
  "run_count": 5,
  "M": 16,
  "efConstruction": 200,
  "ef_search": 40
}
```

---

## 三、典型配置运行案例：PGVector

### 3.1 配置内容

```yaml
float:
  any:
  - base_args: ['@metric']
    constructor: PGVector
    disabled: false
    docker_tag: ann-benchmarks-pgvector
    module: ann_benchmarks.algorithms.pgvector
    name: pgvector
    run_groups:
      M-16:
        arg_groups: [{M: 16, efConstruction: 200}]
        args: {}
        query_args: [[10, 20, 40, 80, 120, 200, 400, 800]]
```

### 3.2 配置解析过程

**假设运行命令**：
```bash
python run.py --dataset glove-100-angular --algorithm pgvector
```

**解析步骤**：

1. **数据集信息**：
   - `point_type = 'float'`
   - `distance = 'angular'`
   - `dimension = 100`

2. **匹配配置**：
   - 查找 `float` → `any` 路径 (因为配置中使用了 `any`)
   - 找到名为 `pgvector` 的算法定义

3. **参数展开**：

   **base_args 替换**：
   ```python
   base_args = ['@metric']
   # 替换后 →
   base_args = ['angular']
   ```

   **arg_groups 展开**：
   ```python
   arg_groups = [{M: 16, efConstruction: 200}]
   # _generate_combinations 处理字典 →
   # 因为字典值不是列表，直接返回 →
   构建参数 = [{M: 16, efConstruction: 200}]
   ```

   **最终构造函数调用**：
   ```python
   PGVector('angular', {M: 16, efConstruction: 200})
   ```

   **query_args 展开**：
   ```python
   query_args = [[10, 20, 40, 80, 120, 200, 400, 800]]
   # 这是单个参数的多个值 →
   查询参数组合 = [10], [20], [40], [80], [120], [200], [400], [800]
   # 共8个查询配置
   ```

4. **生成的 Definition 对象**：
   ```python
   Definition(
       algorithm='pgvector',
       docker_tag='ann-benchmarks-pgvector',
       module='ann_benchmarks.algorithms.pgvector',
       constructor='PGVector',
       arguments=['angular', {'M': 16, 'efConstruction': 200}],
       query_argument_groups=[[10], [20], [40], [80], [120], [200], [400], [800]],
       disabled=False
   )
   ```

### 3.3 实际运行流程

**1. 索引构建阶段** (只执行一次):

```python
algo = PGVector('angular', {'M': 16, 'efConstruction': 200})

# 在 PGVector.__init__ 中:
self._metric = 'angular'
self._m = 16
self._ef_construction = 200
self._query = "SELECT id FROM items ORDER BY embedding <=> %s LIMIT %s"

# 调用 fit()
algo.fit(X_train)  # X_train.shape = (1183514, 100)

# fit() 执行:
# 1. 启动PostgreSQL服务
# 2. 创建向量表: CREATE TABLE items (id int, embedding vector(100))
# 3. 批量插入数据: COPY items FROM STDIN
# 4. 创建HNSW索引:
#    CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops)
#    WITH (m = 16, ef_construction = 200)
# 5. 监控进度并报告
```

**2. 查询阶段** (对每个 ef_search 值执行一次):

```python
# 迭代1: ef_search = 10
algo.set_query_arguments(10)
# 执行: SET hnsw.ef_search = 10

for i in range(5):  # --runs 5，运行5次
    results = []
    for query_vector in X_test:  # 10000个查询
        candidates = algo.query(query_vector, 10)
        # 执行: SELECT id FROM items ORDER BY embedding <=> query_vector LIMIT 10
        results.append(candidates)
    # 记录最佳时间

store_results('glove-100-angular', 10, definition, [10], ...)
# 保存到: results/glove-100-angular/10/pgvector_{"M":16,"efConstruction":200}_10.json

# 迭代2-8: 重复上述过程，ef_search = 20, 40, 80, 120, 200, 400, 800
```

**3. 生成的结果文件** (共8个):

```
results/glove-100-angular/10/
  ├─ pgvector_{"M":16,"efConstruction":200}_10.json
  ├─ pgvector_{"M":16,"efConstruction":200}_20.json
  ├─ pgvector_{"M":16,"efConstruction":200}_40.json
  ├─ pgvector_{"M":16,"efConstruction":200}_80.json
  ├─ pgvector_{"M":16,"efConstruction":200}_120.json
  ├─ pgvector_{"M":16,"efConstruction":200}_200.json
  ├─ pgvector_{"M":16,"efConstruction":200}_400.json
  └─ pgvector_{"M":16,"efConstruction":200}_800.json
```

每个文件内容示例：
```json
{
  "algo": "pgvector",
  "build_time": 125.3,
  "index_size": 524288,
  "best_search_time": 0.0023,
  "M": 16,
  "efConstruction": 200,
  "ef_search": 40,
  "name": "PGVector(m=16, ef_construction=200, ef_search=40)",
  "candidates": 10.0,
  "run_count": 5
}
```

### 3.4 如果修改配置增加参数组合

**假设修改为**：
```yaml
run_groups:
  M-16:
    arg_groups: [{M: [16, 32], efConstruction: [100, 200]}]
    query_args: [[10, 40]]
```

**展开结果**：

构建参数组合：
```python
{M: [16, 32], efConstruction: [100, 200]}
# 笛卡尔积展开 →
[
  {M: 16, efConstruction: 100},
  {M: 16, efConstruction: 200},
  {M: 32, efConstruction: 100},
  {M: 32, efConstruction: 200}
]
```

查询参数组合：
```python
[[10, 40]]
# 展开 →
[10], [40]
```

**生成的 Definition 对象数量**：
- 4个构建配置 × 1个算法 = 4个 Definition
- 每个 Definition 有 2个查询配置

**总共运行**：
- 构建索引：4次 (每个 M/efConstruction 组合一次)
- 查询测试：4 × 2 = 8次
- 结果文件：8个

---

## 四、关键设计特点

### 4.1 参数空间探索

框架通过笛卡尔积自动生成所有参数组合，便于探索参数空间：
- **构建参数**：影响索引质量和构建时间
- **查询参数**：影响查询精度和速度的权衡

### 4.2 结果去重机制

通过文件名哈希避免重复运行：
```python
filename = f"{algorithm}_{json.dumps(params, sort_keys=True)}_{ef_search}.json"
if os.path.exists(filename) and not force:
    skip  # 除非使用 --force
```

### 4.3 Docker隔离

每个算法在独立容器中运行，优点：
- 依赖隔离
- 资源限制（CPU/内存）
- 防止崩溃影响其他任务
- 可重现性

### 4.4 正确性验证

每次查询后重新计算距离，确保返回的索引对应正确的向量：
```python
verified_candidates = [
    (idx, metrics[distance].distance(query, X_train[idx]))
    for idx in candidates
]
```

这能检测索引损坏或实现bug。
