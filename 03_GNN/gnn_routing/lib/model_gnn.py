import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class ShortestPathGNN(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=None, edge_attr_dim=1):
        super().__init__()
        out_dim = out_dim or hidden_dim

        self.conv1 = TransformerConv(
            in_dim,
            hidden_dim,
            edge_dim=edge_attr_dim,
            heads=8,
            dropout=0.2,
            concat=False,
        )
        self.conv2 = TransformerConv(
            hidden_dim,
            out_dim,
            edge_dim=edge_attr_dim,
            heads=8,
            dropout=0.2,
            concat=False,
        )

    def forward(self, x, edge_index, edge_attr):
        x = x.float().contiguous()
        edge_attr = edge_attr.float().contiguous()
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x  # shape: [num_nodes, out_dim]
