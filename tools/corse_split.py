import numpy as np
from math import sqrt
ini = float('inf')
import networkx as nx

def initial_splite(C, graph, id_dict, id_dict_oldtonew, labels, total_score_dict):

    connected_components = list(nx.connected_components(graph))
    new_clusters = []
    for i, component in enumerate(connected_components, start=1):

        subgraph = graph.subgraph(list(component))
        new_node_ids = []
        for old_id in list(component):
            new_node_ids.append(id_dict_oldtonew[old_id])

        component_data = [C[0][0][i] for i in new_node_ids]
        component_score_dict = {node: total_score_dict[node] for node in new_node_ids}
        component_C = [component_data, component_score_dict]

        #select center point
        centers = select_initial_centers(component_C, component_score_dict, subgraph, id_dict, id_dict_oldtonew)

        if len(centers) != 0:
            Distances = []
            for center in centers:
                Distance = nx.single_source_shortest_path_length(subgraph, id_dict[center])
                Distances.append(Distance)

            balls_data_nodes = [[] for _ in centers]
            balls_score_dict = [{} for _ in centers]
            for old_id in list(component):
                min_dist = np.inf
                min_center_idx = -1

                for Distance in Distances:
                    if Distance[old_id] < min_dist:
                        min_dist = Distance[old_id]
                        min_center_idx = next(iter(Distance.keys()))
                center_index = centers.index(id_dict_oldtonew[min_center_idx])
                balls_data_nodes[center_index].append(C[0][0][id_dict_oldtonew[old_id]])

            for index, center in enumerate(centers):
                ball_nodes = balls_data_nodes[index]
                ball_nodes_index = [int(row[-1]) for row in ball_nodes]

                for node in ball_nodes_index:
                    balls_score_dict[index][node] = total_score_dict[node]

            balls = [
                [balls_data_nodes[i], balls_score_dict[i]]
                for i in range(len(centers))
            ]
            for ball in balls:
                new_clusters.append(ball)
    return new_clusters
    

# MY change for selecting centers: we select centers in a class-wise manner, 
# and we also try to avoid selecting neighbors of already selected centers as much as possible. 
# This is done by maintaining a temporary block list that gets reset after each center selection, 
# and only neighbors of the newly chosen center are added to this block list. 
# If we cannot find a valid node under the current block, we clear the block and try again until we have 
# selected the desired number of centers for each class.
def select_initial_centers(component_C, score_dict, subgraph, id_dict, id_dict_oldtonew):
    node_info = component_C[0]
    class_nodes_dict = {}

    for info in node_info:
        label = info[-2]
        node_index = int(info[-1])   # internal/new id
        if label != -1:
            if label not in class_nodes_dict:
                class_nodes_dict[label] = []
            class_nodes_dict[label].append(node_index)

    num_classes = len(class_nodes_dict)
    num_nodes = len(node_info)

    if num_classes != 0:
        centers_per_class = max(1, int(sqrt(num_nodes) / num_classes))
    else:
        return []

    centers = []

    for label, nodes in class_nodes_dict.items():
        node_scores = [(node, score_dict[node]) for node in nodes if node in score_dict]
        node_scores.sort(key=lambda x: x[1], reverse=True)

        selected_for_class = []
        temp_blocked = set()

        while len(selected_for_class) < centers_per_class:
            chosen = None

            for node, _ in node_scores:
                if node in selected_for_class:
                    continue
                if node in temp_blocked:
                    continue
                chosen = node
                break

            if chosen is None:
                # no valid node found under current temporary block
                # clear block and try again
                temp_blocked = set()
                for node, _ in node_scores:
                    if node not in selected_for_class:
                        chosen = node
                        break

            if chosen is None:
                break

            selected_for_class.append(chosen)

            # reset the temporary block
            temp_blocked = set()

            # block only neighbors of the newly chosen center
            original_node = id_dict[chosen]
            for nbr_old in subgraph.neighbors(original_node):
                nbr_new = id_dict_oldtonew[nbr_old]
                if nbr_new not in selected_for_class:
                    temp_blocked.add(nbr_new)

        centers.extend(selected_for_class)

    return centers

