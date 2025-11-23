import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv

class ShortestPathGNNBaseline(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=32, out_dim=None, edge_attr_dim=3):
        super().__init__()
        out_dim = out_dim or hidden_dim

        self.edge_net1 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim * hidden_dim)
        )
        self.edge_net2 = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim * out_dim)
        )

        self.conv1 = NNConv(in_dim, hidden_dim, nn=self.edge_net1, aggr='mean')
        self.conv2 = NNConv(hidden_dim, out_dim, nn=self.edge_net2, aggr='mean')

    def forward(self, x, edge_index, edge_attr):
        x = x.float().contiguous()
        edge_attr = edge_attr.float().contiguous()
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x  # shape: [num_nodes, out_dim]