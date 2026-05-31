def serialize_graph(graph):
    return {
        "nodes": list(graph.nodes()),
        "edges": [
            [source, target]
            for source, target in graph.edges()
        ],
    }


def deserialize_graph(data):
    import networkx as nx

    graph = nx.DiGraph()

    graph.add_nodes_from(
        data.get("nodes", [])
    )

    graph.add_edges_from(
        data.get("edges", [])
    )

    return graph