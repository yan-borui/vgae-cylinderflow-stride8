# 单节点四卡训练手册

## 1. 在正式目标环境准备

使用集群现有的 Linux CUDA 环境和四张已分配的 NVIDIA GPU。推荐 Python 3.11 或更新的项目兼容版本；安装与驱动匹配的 PyTorch、torchvision，然后安装本仓库。依赖下界不是经过验证的环境锁；正式采用的精确版本由目标环境记录。

```bash
git clone https://github.com/yan-borui/vgae-cylinderflow-stride8.git
cd vgae-cylinderflow-stride8
python -m pip install -e .
python -m pip freeze --all > requirements-site.txt

export REPO_DIR="$PWD"
export DATA_DIR="$PWD/data/stride8"
export CAMPAIGN_DIR="$PWD/runs/campaign"
export ENV_PROFILE="$PWD/site-environment.json"
export ACCEPTANCE_DIR="$PWD/runs/acceptance"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 普通服务器按实际分配填写；Slurm 内保留调度器的 CUDA_VISIBLE_DEVICES。
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m vgae_cf prepare --data "$DATA_DIR"
python -m vgae_cf campaign create --root "$CAMPAIGN_DIR"
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --module vgae_cf environment --output "$ENV_PROFILE"
```

环境记录包括 GPU 型号/显存/数量、可见设备、拓扑、驱动、Linux、Python、CUDA/NCCL、PyTorch、PyG 及其可选扩展版本。验收和正式入口逐项核对该记录，且整个 campaign 绑定同一数据与环境。图缓存由 CPU 预处理生成；训练和运行验收使用完整的四卡生产路径。

数据约 1.77 GB，另需图缓存。所有正式 slot 都读取同一份数据与图缓存。Checkpoint 每次评价持久保存完整模型和 Adam 状态，磁盘用量随参数量及评价次数增长；程序不自动清理历史 checkpoint。

`PYTHON` 可指定解释器。`DATA_WORKERS` 默认 0，可按目标环境设置；启用 worker 时采用 spawn，抽帧由显式 seed/epoch/trajectory 确定。

## 2. 同一个入口的缩量验收

```bash
bash scripts/acceptance_4gpu.sh 2
```

保留 slot2 的正式结构、精度、四卡、SyncBatchNorm、优化器和数据格式；仅将每 epoch 曝光量减到 24 图、轮数减到 2、Train/Validation 监测各减到 4 条轨迹。监测仍使用 stored `1..64`。每个 epoch 包含一个全局 16 图 batch 和一个 8 图尾批。

脚本先完成 epoch1、保存后退出，再用同一入口 `--resume` 进入 epoch2，完成评价、权重导出和正常退出。验收输出使用 `mode=acceptance` 和独立目录，不进入 15 次正式结果。该脚本可换 slot 复用，验收较大模型也保留它的正式结构。

发生错误后保留目录，修复实际暴露的问题；在相同源码身份下恢复同一验收任务：

```bash
bash scripts/acceptance_4gpu.sh 2 --resume
# 仅限尚未产生 latest.pt 的初始失败：
bash scripts/acceptance_4gpu.sh 2 --retry-initial
```

源码变化会使旧 campaign 的身份检查失败。需要代码修复时保留旧目录及失败证据，在修订后的代码上重新建立验收 campaign；正式实验的源码修订必须明确记录，不能把改变配方后的结果当作原条件续训。运行验收的当前状态见 [VALIDATION.md](VALIDATION.md)。

## 3. 首轮九个任务

可先运行交点；各任务每次独占四卡。以下命令在同一四卡分配上顺序执行首轮：

```bash
bash scripts/run_4gpu.sh 2
for slot in 0 1 3 4 5 6 7 8; do
    bash scripts/run_4gpu.sh "$slot" || break
done
python -m vgae_cf campaign status --root "$CAMPAIGN_DIR"
```

等候评价时日志持续列出各 rank 的轨迹与 MSE。完成 epoch1 和首次 Train/Validation 评价后，可用该轮耗时估计后续时间；真实早停轮数由 Validation 决定。查看 `status.json`、各 attempt、launcher 日志和 GPU 状态即可，不需要额外监控作业。

续训不占新 slot：

```bash
bash scripts/run_4gpu.sh 2 --resume
```

`latest.pt` 恢复到最近完成的 epoch；中断中的 epoch 从该边界重放。初始失败没有 checkpoint 时可显式使用 `--retry-initial`，仍复用原 slot、原 seed、原配置，并保留失败记录。已完成任务会直接返回，不再次训练。

## 4. 晋级与三 seed 复核

首轮九个任务均有完整结果后：

```bash
python -m vgae_cf campaign confirm --root "$CAMPAIGN_DIR"
for slot in 9 10 11 12 13 14; do
    bash scripts/run_4gpu.sh "$slot" || break
done
python -m vgae_cf report --campaign "$CAMPAIGN_DIR" --output artifacts/report
```

Slot9/10 是基线的 seed1/2，slot11/12 是首轮最好的扩容配置 seed1/2，slot13/14 是第二好的扩容配置 seed1/2。平分按配置 ID 排序。重复执行 `confirm` 返回已冻结的分配，不重新选择或增加任务。首轮尚有失败未恢复时会明确报告缺失结果；不会自动替换候选。

## 5. Slurm

激活环境并导出上述变量，采用能复用同一正式环境记录的节点/GPU 分配。分区、GPU 类型、内存和 wall time 由站点参数指定：

```bash
sbatch --array=0-8 scripts/slurm_array.sh
# 首轮完成并执行 campaign confirm 后：
sbatch --array=9-14 scripts/slurm_array.sh
# 单个失败 slot 沿原任务续跑：
sbatch --array=2 scripts/slurm_array.sh --resume
```

每个 array task 请求一节点、一 task、四张 GPU，再启动四个 torchrun worker。若要控制并发数量，使用 `--array=0-8%2` 等 Slurm 设置；两项同时运行需要八张卡。Slurm 的四卡分配与保存的环境记录应匹配，不能用不同硬件的验收结果替代。

## 6. 单独重做完整 Validation

正常训练结束已自动对选定权重评价全部 Validation100 × stored `1..64`。需要独立重算时指定新的输出目录：

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
    --module vgae_cf evaluate \
    --run "$CAMPAIGN_DIR/runs/w512_d4-4-2_c4_seed0" \
    --data "$DATA_DIR" --environment "$ENV_PROFILE" \
    --output artifacts/center_revaluation
```

该命令读取 `best.json` 指向的权重；它不改变选模结果或任务数量。汇总命令只读取已完成的正式结果，可在安装了绘图依赖的环境运行。
