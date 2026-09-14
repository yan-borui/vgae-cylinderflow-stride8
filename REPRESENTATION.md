# 后续 DiT 的表示接口

每个已完成 run 提供 `codec.pt` 和 `codec_metadata.json`。这份权重来自 Validation24 选中的 checkpoint，包含 SyncBatchNorm 的 running mean/variance。附带 architecture、decoder 深度、Train 归一化、图层级、节点顺序、条件定义和选择记录。

```python
import torch
from vgae_cf.data import Dataset
from vgae_cf.model import UVCodec

data = Dataset("data/stride8")
codec = UVCodec("runs/campaign/runs/w512_d4-4-2_c4_seed0/codec.pt", device="cuda")
graph = data.static_graph(1000)
uv = torch.from_numpy(data.read_uv(1000, 1, 2)[0])  # physical [N, 2]
posterior = codec.encode(graph, uv)
latent = posterior.mean                         # [N3, C]
reconstruction = codec.decode(graph, latent, posterior.context)  # physical [N, 2]

# 对 DiT 预测的 [N3, C] latent，可只根据静态图重新生成 condition context：
# prediction = codec.decode(graph, predicted_latent)
```

`posterior` 包含 `sample`、`mean`、`logvar` 和可复用的静态 decoder context。编码时自动使用 Train UV 归一化，解码时返回物理 UV；latent 没有额外标准化。模型内部训练使用采样，重建比较使用 `mean`。若后续 DiT 需要 latent 标准化，应在新表示的 Train 编码上独立拟合并记录。

|对象|含义|
|---|---|
|`graph.pos`|细图节点坐标 `[N,2]`|
|`graph.pos_3`|固定第三层粗图坐标 `[N3,2]`|
|`graph.edge_index_3`|粗图有向邻接 `[2,E3]`|
|latent|与 `pos_3` 同一节点顺序的 `[N3,C]`，每帧一个表示|
|条件节点特征|标准化 inlet speed + 三个 node-type one-hot；共4通道|
|边特征|原方式缩放的二维相对位置|
|`boundary_values`|stored frame0 的标准化 inlet/wall UV，内部节点零占位|

不同网格的 N、N3 可不同；对相同轨迹，九个配置使用相同的图缓存，只有 C 和网络容量变化。宽度不是 token 数。三个模块深度均在已有图层级内取值，深度线不增加粗化层级。

调用者传入由 `Dataset.static_graph` 生成的完整层级图；批量处理使用随仓库提供的 `dgn4cfd.loader.Collater`。该 collater 会就地修正跨图索引，因此 Dataset 返回独立 clone，接口内部也 clone 图。Decoder context 只适用于生成它的权重、图、设备和节点顺序；重新编码静态条件可支持未携带 context 的 DiT 预测。时间轴由调用者组织，这里没有时间压缩模块。
