# VGAE · CylinderFlow stride-8 · 四卡扩容

用 **15 次正式训练**检验 UV VGAE 能否通过扩大网络和 latent 通道降低重建误差。首轮 9 个配置共享训练配方；三条扩容线交于 **width512 / encoder [4,4,2] / latent4**。交点是先验候选，胜出配置由 Validation 决定。

**交付状态：源码、启动器和两阶段管理已实现；目标四卡运行验收待完成。当前没有训练结果或已验证的显存上限。** 开始训练请读 [FOUR_GPU.md](FOUR_GPU.md)，验收范围见 [VALIDATION.md](VALIDATION.md)。

## 固定的九个配置

|首轮 slot|配置 ID|宽度|encoder 深度|latent 通道|所属线|
|---|---|---:|---|---:|---|
|0|baseline|126|[2,2,1]|1|UV 原尺度基线|
|1|w256_d4-4-2_c4|256|[4,4,2]|4|宽度|
|2|w512_d4-4-2_c4|512|[4,4,2]|4|三线交点|
|3|w1024_d4-4-2_c4|1024|[4,4,2]|4|宽度|
|4|w512_d2-2-1_c4|512|[2,2,1]|4|深度|
|5|w512_d8-8-4_c4|512|[8,8,4]|4|深度|
|6|w512_d4-4-2_c1|512|[4,4,2]|1|latent|
|7|w512_d4-4-2_c2|512|[4,4,2]|2|latent|
|8|w512_d4-4-2_c8|512|[4,4,2]|8|latent|

定义在 [scale_axes.json](vgae_cf/configs/scale_axes.json)，`campaign create` 将其展开成完整的 architecture、recipe 和 slot 清单。Decoder 深度始终为 encoder 的逆序；condition encoder 深度固定 `[1,1,1]`。宽度共同作用于三个模块。预处理保留原来的五层图，编码只到第三层；三条线均保留第三层节点、坐标、邻接、池化和节点顺序。

首轮所有配置从随机初始化训练 seed0。按 Validation-24 物理 UV MSE 选出两种扩容配置，与基线各补训 seed1、seed2，形成另外 6 个 slot。交点只占一个首轮 slot；管理入口不会添加第 16 个正式 slot。完整协议和统计口径见 [PROTOCOL.md](PROTOCOL.md)。

## 入口

```bash
git clone https://github.com/yan-borui/vgae-cylinderflow-stride8.git
cd vgae-cylinderflow-stride8
python -m pip install -e .
python -m vgae_cf prepare --data data/stride8
python -m vgae_cf campaign create --root runs/campaign
```

准备目标环境、设置 `DATA_DIR` / `CAMPAIGN_DIR` / `ENV_PROFILE` 后：

```bash
bash scripts/acceptance_4gpu.sh 2     # 同一正式模型和四卡拓扑的缩量验收
bash scripts/run_4gpu.sh 2            # 优先候选，正式 seed0
```

每个任务独占单节点四张 GPU，每卡 batch4，全局 batch16；1000 条 Train 轨迹产生 62 个完整 batch 和一个真实的 8 图尾批，每卡尾批 2 图。使用 FP32、SyncBatchNorm、Adam、LR `1e-4`、clip1、dropout0.1、KL `1e-6`，关闭 AMP 和 TF32。其余 8 个 slot 及 Slurm 命令见 [FOUR_GPU.md](FOUR_GPU.md)。

## 数据、结果与表示接口

数据来自 [固定 revision 的 Train/Validation 发布](https://huggingface.co/datasets/DingDong1921/mgn-cylinderflow-stride8-75frames/tree/8eae2c7a697e7d01f3b98f4d642ea476784df84a)。准备工具下载 HDF5 和 manifest，读取 UV 两个通道，沿用 Train 归一化并缓存图层级；Test 保持封存。

训练结果保留滚动恢复点、每次选模评价的完整 checkpoint、最佳索引、逐轨迹指标、Validation-100 重建场、共同色标图及 `codec.pt`。后续 DiT 使用 `UVCodec.encode/decode`，接口和张量含义见 [REPRESENTATION.md](REPRESENTATION.md)。

```bash
python -m vgae_cf campaign status --root runs/campaign
python -m vgae_cf report --campaign runs/campaign --output artifacts/report
```

报告提供首轮三条曲线、参数量/latent 元素数/曝光量/GPU 时间曲线、三 seed 均值和样本标准差、同 seed 的基线配对变化及辅助物理指标。只有完成三 seed 复核才给出最终配置；三条单轴曲线提供交点附近的证据。

源码采用 Apache-2.0；上游代码及修改说明见 [NOTICE](NOTICE)。数据遵循其原发布条款。
