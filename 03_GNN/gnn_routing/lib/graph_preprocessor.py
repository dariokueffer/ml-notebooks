import numpy as np
from torch_geometric.utils import from_networkx
import torch
import torch_geometric.data as pyg_data

import networkx as nx


class GraphPreprocessor:
    def __init__(self, graph):
        self.graph = self._add_super_node(graph)
        self.feature_matrix = self._create_node_feature_matrix()
        self.edge_index, self.edge_attrs = self._create_edges_and_weights()

    def _add_super_node(self, G):
        G = G.copy()

        super_node = max(G.nodes()) + 1
        G.add_node(super_node, pos=(0.5, 0.5))

        original_nodes = list(G.nodes())[:-1]

        for node in original_nodes:
            for direction in [(super_node, node), (node, super_node)]:
                u, v = direction
                G.add_edge(u, v, weight=0.1)
                G[u][v]["distance"] = 0.1

        return G

    def _create_node_feature_matrix(self):
        NUMBER_OF_FEATURES = 3  # x, y, is_super_node
        num_nodes = len(self.graph.nodes())
        feature_matrix = np.zeros((num_nodes, NUMBER_OF_FEATURES))

        for i, node in enumerate(self.graph.nodes()):

            x, y = self.graph.nodes[node]["pos"]

            super_node = max(self.graph.nodes())
            is_super_node = int(node == super_node)

            feature_matrix[i] = [x, y, is_super_node]

        return feature_matrix

    def _create_edges_and_weights(self):
        edges = []
        edge_attrs = []
        for u, v in self.graph.edges():
            for src, dst in [(u, v), (v, u)]:
                edges.append((src, dst))
                data = self.graph[u][v]
                distance = data["distance"]
                edge_attrs.append([distance])
        return np.array(edges), np.array(edge_attrs)

    def create_pyg_data(self):
        edge_index = torch.tensor(self.edge_index.T, dtype=torch.long)
        edge_attr = torch.tensor(self.edge_attrs, dtype=torch.float)
        print("Edge attr shape:", edge_attr.shape)
        data = pyg_data.Data(
            x=torch.tensor(self.feature_matrix, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        return data

    def adjacency_from_nx(self):
        nodelist = list(self.graph.nodes())
        A = nx.to_numpy_array(self.graph, nodelist=nodelist)
        A = torch.tensor(A, dtype=torch.bool)
        return A

    def distance_matrix_from_nx(self):
        num_nodes = len(self.graph.nodes())
        D = torch.full((num_nodes, num_nodes), float("inf"))

        for u, v, data in self.graph.edges(data=True):
            weight = data.get("distance", 1.0)
            D[u, v] = weight
            D[v, u] = weight

        return D
