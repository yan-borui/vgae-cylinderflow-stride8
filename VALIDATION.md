# 交付检查与目标环境验收

记录日期：2026-09-14。

## 本地静态检查

检查在 Windows 上进行，不导入或执行训练模型。Python 语法/配置解析、Ruff F 检查、格式检查、三个 shell 启动器语法检查及 Git 空白检查均通过。

```bash
python scripts/static_check.py
ruff check . --select F
ruff format --check .
bash -n scripts/run_4gpu.sh
bash -n scripts/acceptance_4gpu.sh
bash -n scripts/slurm_array.sh
git diff --check
```

`static_check.py` 只解析/编译语法、读取 JSON/TOML 和检查配置数量，不执行编译后的应用代码。静态检查不能证明 CUDA 运算、显存容量、DDP 同步、恢复或实验质量。

## 唯一缩量运行验收流程

入口为 [scripts/acceptance_4gpu.sh](scripts/acceptance_4gpu.sh)，调用与正式训练相同的 `python -m vgae_cf train`。具体命令见 [FOUR_GPU.md](FOUR_GPU.md)。只能在与正式运行匹配的 Linux、四张目标 GPU、拓扑、CUDA/NCCL、PyTorch/PyG 及依赖版本下执行。

当前运行验收状态：**未执行**。本地 Windows 环境与正式 Linux 四卡环境不匹配，且尚无目标硬件的验证记录；没有使用 CPU 或单 GPU 替代测试，也没有执行额外测试套件。

在目标环境运行时，保留两次启动的完整日志、rank 异常、退出码和产物，确认：

1. 四个 rank 使用注册的目标 GPU 与 NCCL；数据和静态图加载成功。
2. 使用注册的正式模型、FP32 和 SyncBatchNorm，完成全局16图及8图尾批的前向、反向与更新。
3. Epoch1 完成指定 Train/Validation 评价，产生持久 checkpoint、rolling checkpoint 与 best 索引。
4. 第二次启动从同一 slot 的 rolling checkpoint 恢复各 rank RNG、BN、Adam、调度/早停计数器与数据游标，继续到 epoch2。
5. 产出有限且覆盖完整预定样本的指标、共享色标图、最终权重和 codec 元数据；两次启动均正常退出。

该流程验证其实际覆盖的启动、训练、评价、保存、恢复和结束路径。目标最大图/最大模型显存、500–5000 epochs 稳定性、所有候选的运行情况及科学收益仍需正式运行证据。失败留在原任务记录中，按暴露的具体错误修复；不自动增加第16个正式训练。
