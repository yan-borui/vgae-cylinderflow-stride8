# CylinderFlow Test 评测

本分支支持官方 Test100 的下载、独立转换和评测。使用已有环境，从仓库根目录执行。将下面的路径变量替换为对应已训练运行的路径，`OUT` 使用全新结果目录。

## 下载与转换

下载脚本只依赖 Python 标准库，支持断点续传和共享目录锁。两个数据集原始 Test 合计约 6.41 GB；下载不需要模型权重。

```bash
export RAW_TEST_ROOT=/shared/data/mgn-test-raw
bash scripts/download_test.sh --dataset both --raw-root "$RAW_TEST_ROOT"
```

官方来源：[Airfoil Test](https://storage.googleapis.com/dm-meshgraphnets/airfoil/test.tfrecord)、[CylinderFlow Test](https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord)，以及对应目录的 `meta.json`；与 [MeshGraphNets 官方下载入口](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh) 一致。

本分支评测 **CylinderFlow**。原始 Test 文件可以跨方法共享；转换目录绑定训练 manifest 的归一化，统计不同的运行使用各自目录。

```bash
export TRAIN_MANIFEST=/absolute/path/to/original_training_manifest.json
export TEST_DIR=/shared/data/cylinderflow-test/selected-normalization/test65
python -m mgn_data.prepare_test \
  --dataset cylinderflow --raw-dir "$RAW_TEST_ROOT/cylinder_flow" \
  --train-manifest "$TRAIN_MANIFEST" --output-dir "$TEST_DIR"
```

转换仅保留 raw 0, 8, …, 512，共 65 帧 UVP。Test 保持独立的 100 条官方轨迹，manifest 的 split 为 `test`。归一化逐项复制对应 Train manifest；已有相同完整转换可复用，失败转换保留在独立临时目录。转换检查和转换所需依赖来自已有 NumPy/h5py 环境。

## 使用选定权重

```bash
export OUT=/absolute/path/to/new_test100_result
torchrun --standalone --nproc_per_node=4 -m tools.evaluate_test \
  --run "$RUN" --data "$TRAIN_DATA_DIR" --environment "$ENVIRONMENT" \
  --test-data-dir "$TEST_DIR" --output "$OUT"
```

`RUN` 包含原有 `best.json`；该索引按 Validation24 的 UVP total loss 选择。`TRAIN_DATA_DIR`、`ENVIRONMENT` 使用原正式运行的数据目录和环境记录。入口核对原模型源码、数据、环境和选择记录，沿用四进程 CUDA/NCCL 及 posterior mean。

VGAE 对 Test100 的未来 64 帧逐帧编码重建，输出重建指标，报告时单列为表示重建。结果含 `evaluation_identity.json`、`summary.json`、逐轨迹 JSON/NPZ 和代表图。

## 比较口径与验证范围

CylinderFlow 的物理帧间隔为 0.08，条件为帧 0，预测和评分覆盖后续 64 帧。沿用模型的原生边界处理、推理和指标。Train 统计、原权重选择记录以及已经报告的 Validation 结果分别保留；补测结果标明 Test、采样预算、聚合方式和失败数。历史权重的 Train75 或前 65 帧训练暴露继续按原运行记录报告。

本次入口交付仅完成静态语法、引用和 diff 检查。正式 GPU 推理及完整 Test 转换尚待目标环境执行；本次代码更新没有产生新的 Test 指标。
