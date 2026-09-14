"""Graph tensor container; plotting lives in the experiment report module."""

from torch_geometric.data import Data


class Graph(Data):
    """Keep the upstream graph and batching semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "batch" not in kwargs:
            self.batch = None
