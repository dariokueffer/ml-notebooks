import numpy as np
import networkx as nx
from scipy.spatial import distance


class GraphGenerator:
    def __init__(self, N, seed=42):
        self.nodes = self.generate_nodes(N, seed)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(N))

    def generate_nodes(self, N, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.rand(N, 2)

    def random_geometric_graph(self, radius):
        G = nx.Graph()
        G.add_nodes_from(range(len(self.nodes)))

        for i in range(len(self.nodes)):
            G.nodes[i]["pos"] = self.nodes[i]

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if np.linalg.norm(self.nodes[i] - self.nodes[j]) < radius:
                    G.add_edge(i, j)
        G = self.add_edge_features(G)
        return G

    def relative_neighborhood_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(len(self.nodes)))

        for i in range(len(self.nodes)):
            G.nodes[i]["pos"] = self.nodes[i]

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist_ij = np.linalg.norm(self.nodes[i] - self.nodes[j])
                is_rho = True
                for k in range(len(self.nodes)):
                    if k != i and k != j:
                        dist_ik = np.linalg.norm(self.nodes[i] - self.nodes[k])
                        dist_jk = np.linalg.norm(self.nodes[j] - self.nodes[k])
                        if dist_ik < dist_ij and dist_jk < dist_ij:
                            is_rho = False
                            break
                if is_rho:
                    G.add_edge(i, j)
        G = self.add_edge_features(G)
        return G

    def gabriel_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(len(self.nodes)))

        # Assign positions
        for i in range(len(self.nodes)):
            G.nodes[i]["pos"] = self.nodes[i]

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                midpoint = (self.nodes[i] + self.nodes[j]) / 2
                radius_sq = np.sum((self.nodes[i] - midpoint) ** 2)
                is_gabriel = True

                for k in range(len(self.nodes)):
                    if k != i and k != j:
                        dist_sq = np.sum((self.nodes[k] - midpoint) ** 2)
                        if dist_sq < radius_sq:
                            is_gabriel = False
                            break

                if is_gabriel:
                    G.add_edge(i, j)

        G = self.add_edge_features(G)
        return G

    # Adds Distance and Speed Features
    # Will need to add highway/bicycle road features
    def add_edge_features(self, G):
        for u, v in G.edges():
            pos_u = G.nodes[u]["pos"]
            pos_v = G.nodes[v]["pos"]
            G[u][v]["distance"] = np.linalg.norm(np.array(pos_u) - np.array(pos_v))

        return G
