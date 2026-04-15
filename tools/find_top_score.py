ini = float('inf')
from operator import itemgetter

def find_max_degree(degree_dict):
    sorted_dict_desc = dict(sorted(degree_dict.items(), key=itemgetter(1), reverse=True))
    max_index = [list(sorted_dict_desc.keys())[0], list(sorted_dict_desc.keys())[1]]
    max_value = 0
    return max_value, max_index

# new change i did 
def find_top_two_nodes(score_dict):
    sorted_dict_desc = dict(sorted(score_dict.items(), key=itemgetter(1), reverse=True))
    max_index = [list(sorted_dict_desc.keys())[0], list(sorted_dict_desc.keys())[1]]
    return 0, max_index