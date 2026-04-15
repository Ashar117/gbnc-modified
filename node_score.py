import networkx as nx

def get_global_score_dict(graph, method='degree'):
    if method == 'degree':
        return dict(graph.degree())
    elif method == 'pagerank':
        return nx.pagerank(graph)
    else:
        raise ValueError(f"Unsupported score method: {method}")

def remap_score_dict(score_dict_old):
    score_dict = {}
    for new_id, (_, score) in enumerate(score_dict_old.items()):
        score_dict[new_id] = score
    return score_dict