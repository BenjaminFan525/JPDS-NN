import numpy as np
import networkx as nx
import pickle

def angle_diff(ang1, ang2):
    diff = ang1 - ang2
    if diff < - np.pi:
        return diff + 2 * np.pi
    elif diff >= np.pi:
        return diff - 2 * np.pi
    return diff

def gen_node(type, field, idx):
    return type + "-" + str(field) + '-' + str(int(idx))

def get_euler_dis(graph: nx.Graph, node1, node2):
        return np.linalg.norm(graph.nodes[node1]['coord'] - graph.nodes[node2]['coord'])

def gen_edge(graph: nx.Graph, node1, node2):
    return (node1, node2, get_euler_dis(graph, node1, node2))

def get_dijkstra_path(graph: nx.Graph, node1, node2, cartesian = False):
    '''
    Return a dijkstra path in the graph, from node1 to node2.
    If cartesian is True, return the coordinates of the nodes, otherwise the nodes' names.
    '''
    if isinstance(node1, tuple):
        node1 = gen_node(*node1)
    if isinstance(node2, tuple):
        node2 = gen_node(*node2)
    node_path = nx.dijkstra_path(graph, node1, node2)
    if not cartesian:
        return node_path
    return [graph.nodes[node]['coord'] for node in node_path]     

def get_dijkstra_path_length(graph: nx.Graph, node1, node2):
    if isinstance(node1, tuple):
        node1 = gen_node(*node1)
    if isinstance(node2, tuple):
        node2 = gen_node(*node2)
    return nx.dijkstra_path_length(graph, node1, node2)

def get_point(line: np.ndarray, split):
    '''
    Get a point on a line.
    line: a 2 x 2 NDArray, the 2 ends of a line
    split: the split on the line, varies from 0 to 1, 0 is the end line[0], 1 is line[1]
    '''
    return line[0] + split * (line[1] - line[0])


def calc_abc_from_line_2d(line):
    a = line[0][1] - line[1][1]
    b = line[1][0] - line[0][0]
    c = line[0][0] * line[1][1] - line[1][0] * line[0][1]
    return a, b, c


def get_line_cross_point(line1, line2):
    a0, b0, c0 = calc_abc_from_line_2d(line1)
    a1, b1, c1 = calc_abc_from_line_2d(line2)
    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return np.array([x, y])

def save_dict(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_dict(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

COLORS = (
    [
        # deepmind style
        'red',
        'yellow',
        'black',
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        
        'pink',
        'cyan',
        'magenta',
        
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)