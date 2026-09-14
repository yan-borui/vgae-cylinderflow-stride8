# 训练、评价与证据口径

## 数据和网络

Train 使用固定的 1000 条轨迹，Validation 使用固定的 100 条；只读取发布文件中的 UV。每条轨迹 75 个 stored frames，raw 对应 `0,8,...,592`，stored 间隔 0.08。Test 不在允许访问的索引集合中。

沿用 manifest 的 Train-only affine normalization：UV mean `[0.5661443793145688, 0.0006225839965382098]`、std `[0.57014025360096, 0.1436773068500826]`；条件中的 inlet speed 也用 Train 统计量归一化。统计量不从 Validation 重新估计。固定版本及完整 manifest 身份保存在运行元数据中。

每 epoch 每条 Train 轨迹抽取一帧 stored `0..74`。抽帧使用 `NumPy SeedSequence([seed, epoch, trajectory])`；排列使用独立的 `SeedSequence([seed, epoch, 1100])`，不会被模型初始化、dropout 或 latent 采样改变。相同 seed 的所有配置具有相同抽帧与 batch 序列。

VGAE 的 condition encoder / field encoder / decoder 均使用配置宽度；SELU、sum aggregation、两层 FNN、dropout0.1 沿用上游。Condition encoder 深度 `[1,1,1]`，其节点/边 BatchNorm 转为 SyncBatchNorm，保留 affine=False、momentum=0.001。Latent 不额外标准化。Field encoder 深度为配置值，decoder 为逆序。

## 四卡重建和 KL

对每张图先计算物理 UV MSE，再等权平均。令第 $g$ 张图有 $N_g$ 个细图节点，$sigma_c$ 是 Train UV 标准差，$hat{x},x$ 是归一化预测与目标：

$$
r_g=\frac{1}{2N_g}\sum_{i=1}^{N_g}\sum_{c\in\{u,v\}}[\sigma_c(\hat{x}_{gic}-x_{gic})]^2,
\quad L_{rec}=\frac{1}{G}\sum_g r_g.
$$

设整个四卡 batch 共有 $M=\sum_g N_{3,g}C$ 个 latent 元素：

$$
L_{KL}=-\frac{1}{2M}\sum_{j=1}^{M}(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2),
\quad L=L_{rec}+10^{-6}L_{KL}.
$$

每个 rank 反传 $4(\sum_{g\in rank}r_g/G+10^{-6}\sum_{j\in rank}KL_j/M)$；$G,M$ 来自跨 rank 求和，抵消 DDP 的梯度平均。节点数不同的图具有相同重建权重；KL 保持逐 latent 元素平均。尾批实际为 $G=8$，每卡两图，没有 padding、丢弃或重复。参考 [DDP 的梯度归约约定](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html)；[SyncBatchNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SyncBatchNorm.html)采用每进程一 GPU 的方式。

训练采样 posterior；评价使用 posterior mean。Inlet 和 wall 上的 UV 使用 stored frame0 的已知边界值写回，Train 与所有评价一致。边界输出同时记录写回前的 raw decoder 误差，避免把硬约束的结果误认为学到的边界质量。Outlet 未强制写回。

## 配方和选模

FP32，无 AMP/TF32；Adam `lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0`，梯度范数 clip1，无梯度累积。Epoch1 和每 10 epochs，在固定 Train24 与 Validation24 上评价 stored `1..64`。24 条轨迹从各 split 按等距位置、四舍五入选择，保留索引到每次评价文件。Eval 模式冻结 BN/dropout，评价前后恢复各 rank RNG。

选模 MSE 对每条轨迹等权，在轨迹内部对 64 帧、节点和两个 UV 分量等权；单位为物理速度平方。Train24 使用完全相同口径；日志里的 `train_sample_uv_mse` 来自训练模式和随机单帧，仅用于优化诊断。

Validation24 MSE 严格变小才更新 best，平分保留较早 checkpoint。每连续 **5 次**有效评价未改善，LR ×0.1，下限 `1e-8`。独立累计的 early-stop 计数不因 LR 下降清零。最少500、最多5000 epochs；满足最少轮数且连续20次有效评价无严格改善后停止。调度状态显式保存，避免把 epoch 次数当作评价次数。

每个 run 结束后，用其选定权重补充 Validation100 × 64 帧。晋级根据 Validation24；最终三个复核配置按三 seed 平均 Validation24 MSE 选择。Validation100 及下列指标用于补充判断与披露，不参与 LR、早停或晋级：

|指标|定义|
|---|---|
|area_uv_relative_rmse|三角形面积的三分之一分配给各节点；整个序列误差能量与真值能量之比开根号，再对轨迹等权平均|
|vorticity_rmse|三角形分片线性梯度的 $\partial_xv-\partial_yu$，误差按三角形面积与时间加权 RMS|
|divergence_rmse|$\partial_xu+\partial_yv$ 的面积/时间加权误差 RMS|
|predicted/reference_divergence_rms|分别保留预测与参考离散散度本身的 RMS|
|边界 UV RMSE|inlet、wall、outlet、全部边界分别计算；保留写回前/后两个版本|

无效指标保留空值及有效计数；选模 MSE 必须覆盖所有预定轨迹且有限，任何无效轨迹使该次评价失败。不得跳过坏样本后报告均值。重建图固定物理 UV 色标，最终报告对基线/候选/三 seed 使用共同色标，并展示由基线误差选定的最好、中位、最差轨迹。

## 两阶段及成本

首轮9次 seed0 仅作筛选。两种扩容候选和基线各补 seed1、seed2，总计15个正式 run 身份。失败记录和续训使用原身份；任务列表不会因失败扩张。

报告三 seed 的均值、样本标准差（ddof=1）和逐 seed 的 `candidate − baseline` MSE，负值表示改善。首轮筛选存在选择偏差，只有3个 seed，不能据此保证统计显著性。三条单轴曲线描述交点附近的变化，不估计三轴交互或保证全局最优。

同时报告参数量、第三层节点数 × latent 通道、完成的帧曝光数、optimizer steps、训练 GPU 时间、包括评价在内的 committed GPU 时间、各 rank 峰值已分配显存，以及同步后的单图 encoder/decoder 毫秒数。Encoder 计时包括 condition encoder；不含 I/O。成本曲线使用实际曝光量和训练 GPU 时间，便于区分容量变化与更长训练带来的收益。崩溃前未提交的工作量单列在 attempt/launcher 日志，不能冒充精确总成本。

## 恢复与文件事务

每个完整 epoch 原子替换 `latest.pt`；每次 Train24/Validation24 评价另存一份完整 checkpoint，`best.json` 指向其中最佳项。状态包含模型（含 BN）、Adam、LR 与计数器、early-stop、已完成 epochs/updates/曝光量、数据游标、Python/NumPy/Torch CPU/对应 GPU 的每 rank RNG、每 rank buffers，以及数据、源码和环境记录。

持久 checkpoint、rolling checkpoint 和 best 索引通过临时文件替换发布；目录进程锁阻止同一个 run 并发训练。每次评价和启动使用独立 attempt 目录，保留失败及重复执行的证据。若在某 epoch 中断，恢复最近完成边界并重放该 epoch；恢复随机状态不构成 CUDA scatter 运算逐 bit 一致性的保证。结束训练后的完整 Validation 中断，可从 rolling checkpoint 的 `final_evaluation` 状态恢复。
