import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.lines as mlines


def plot_graph(G, shortest_path=None, title=None):
    """
    Draws an undirected geometric graph
    • normal road  → solid grey
    • highway      → dotted grey
    • bicycle path → dashed green
    • edges on `shortest_path` → red (same style as their road_type)
    """
    pos_raw = nx.get_node_attributes(G, "pos")
    pos = {n: (float(x), float(y)) for n, (x, y) in pos_raw.items()}

    plt.figure(figsize=(8, 8))

    # Split edges by type
    street_edges, highway_edges, bike_edges, path_edges = [], [], [], []

    path_edge_set = set()
    if shortest_path:
        path_edge_set.update(zip(shortest_path, shortest_path[1:]))
        path_edge_set.update(zip(shortest_path[1:], shortest_path))

    for u, v in G.edges():
        if (u, v) in path_edge_set:
            path_edges.append((u, v))
            continue

        etype = G[u][v].get("road_type", "normal road")
        if etype == "highway":
            highway_edges.append((u, v))
        elif etype == "bicycle path":
            bike_edges.append((u, v))
        else:
            street_edges.append((u, v))

    # Draw non-path edges
    if street_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=street_edges, edge_color="grey", width=1.5, style="solid"
        )
    if highway_edges:
        coll = nx.draw_networkx_edges(
            G, pos, edgelist=highway_edges, edge_color="grey", width=1.5
        )
        coll.set_linestyle("dotted")
    if bike_edges:
        coll = nx.draw_networkx_edges(
            G, pos, edgelist=bike_edges, edge_color="green", width=1.5
        )
        coll.set_linestyle("dashed")

    # Draw (highlighted) shortest-path edges
    for u, v in path_edges:
        etype = G[u][v].get("road_type", "normal road")
        if etype == "highway":
            style = "dotted"
            color = "red"
        elif etype == "bicycle path":
            style = "dashed"
            color = "orange"
        else:
            style = "solid"
            color = "red"
        coll = nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], edge_color=color, width=2.5
        )
        coll.set_linestyle(style)

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color="blue")
    nx.draw_networkx_labels(
        G,
        pos,
        labels={node: str(node) for node in G.nodes()},
        font_size=6,
        font_color="white",
    )

    plt.plot([0, 0, 1, 1, 0], [0, 1, 1, 0, 0], color="black")

    n, m = G.number_of_nodes(), G.number_of_edges()
    plt.title(f"{title or 'Graph'}  —  nodes: {n}, edges: {m}")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis("off")

    # Legend
    handles = [
        mlines.Line2D([], [], color="grey", linestyle="solid", lw=2, label="Street"),
        mlines.Line2D([], [], color="grey", linestyle="dotted", lw=2, label="Highway"),
        mlines.Line2D(
            [], [], color="green", linestyle="dashed", lw=2, label="Bicycle path"
        ),
        mlines.Line2D(
            [], [], color="red", linestyle="solid", lw=2, label="Shortest path (street)"
        ),
        mlines.Line2D(
            [],
            [],
            color="red",
            linestyle="dotted",
            lw=2,
            label="Shortest path (highway)",
        ),
        mlines.Line2D(
            [],
            [],
            color="orange",
            linestyle="dashed",
            lw=2,
            label="Shortest path (bicycle path)",
        ),
    ]
    plt.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()

    # optionally: drop very short paths


def drop_pair(path, number_of_nodes):
    nodes_to_min_len_mapping = {
        10: 2,
        50: 3,
        100: 4,
    }
    try:
        threshold = max(k for k in nodes_to_min_len_mapping if k <= number_of_nodes)
        min_len = nodes_to_min_len_mapping[threshold]
    except ValueError:
        min_len = 2
    return (len(path) - 2) < min_len  # subtract source and target nodes
