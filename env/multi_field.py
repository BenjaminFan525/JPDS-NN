from env.field import Field
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, List, Optional
from geopandas import GeoSeries
from shapely import geometry
import utm
import json
import networkx as nx
import utils.common as ucommon
import random
import torch
import copy
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

class multiField():
    defauts = {
        "cfg": "",
        "min_bound": 0.2,
        "working_width_range": [12, 24]
    }

    def __init__(self, num_splits, type = '#', width: Tuple = (200, 300), center_lat_lon = (40, 116), working_width = None, 
                 boundary_coords: Optional[np.ndarray] = None, starts: Optional[str] = None, ends: Optional[str] = None, 
                 num_starts=None, num_ends=None, num_veh=None):
        for key, val in self.defauts.items():
            setattr(self, key, val)
        
        # create multi field boundry
        self.num_splits = num_splits
        self.boundary_coords = boundary_coords
        if self.boundary_coords is None:
            split = []
            left = 0
            right = 1
            for idx in range(4):
                if idx == 3:
                    if split[0] > 1 - self.min_bound:
                        left = np.sqrt(self.min_bound ** 2 - (1 - split[0]) ** 2)
                split.append((right - left) * np.random.random() + left)
                
                if split[-1] < self.min_bound:
                    right = 1 - np.sqrt(self.min_bound ** 2 - split[-1] ** 2)
            
            self.boundary_coords = np.array(
                [
                    [split[0], 0],
                    [0, 1 - split[1]],
                    [1 - split[2], 1],
                    [1, split[3]]
                ]
            )

            h = (width[1] - width[0]) * np.random.random() + width[0]
            w = (width[1] - width[0]) * np.random.random() + width[0]
            self.boundary_coords[:, 0] *= w
            self.boundary_coords[:, 0] -= w / 2
            self.boundary_coords[:, 1] *= h
            self.boundary_coords[:, 1] -= h / 2

        self.boundary = np.array([self.boundary_coords, np.roll(self.boundary_coords, -1, axis=0)]).transpose(1, 0, 2)

        # generate fields
        if type == '#':
            upper_points = [self.boundary_coords[1]]
            lower_points = [self.boundary_coords[0]]
            for r in range(num_splits[1]):
                split_temp = 0.5 * np.random.random() + 1 / 2 / (num_splits[1] - r + 1)
                upper_points.append(ucommon.get_point(np.array([upper_points[-1], self.boundary_coords[2]]), split_temp))
                split_temp = 0.5 * np.random.random() + 1 / 2 / (num_splits[1] - r + 1)
                lower_points.append(ucommon.get_point(np.array([lower_points[-1], self.boundary_coords[3]]), split_temp))
            upper_points.append(self.boundary_coords[2])
            lower_points.append(self.boundary_coords[3])
            self.fields_coords = []
            self.Graph = nx.Graph()
            self.get_fields(upper_points, lower_points, num_splits[0])
        elif type == 'T':
            split_temp = 0.6 * np.random.random() + 0.2
            p1 = ucommon.get_point(self.boundary[1], split_temp)
            split_temp = 0.6 * np.random.random() + 0.2
            p2 = ucommon.get_point(self.boundary[3], split_temp)
            split_temp = 0.6 * np.random.random() + 0.2
            p3 = ucommon.get_point(self.boundary[0], split_temp)
            split_temp = 0.6 * np.random.random() + 0.2
            p4 = ucommon.get_point(np.array([p1, p2]), split_temp)
            self.fields_coords = [
                [
                    self.boundary_coords[1],
                    p3,
                    p4,
                    p1,
                    self.boundary_coords[1]
                ],
                [
                    p1,
                    p2,
                    self.boundary_coords[3],
                    self.boundary_coords[2],
                    p1
                ],
                [
                    p3,
                    self.boundary_coords[0],
                    p2,
                    p4,
                    p3
                ]
            ]
            self.Graph = nx.Graph()
            for field_num, points in enumerate(self.fields_coords):
                self.Graph.add_nodes_from([(ucommon.gen_node('bound', field_num, idx), {'coord': point}) for idx, point in enumerate(points[:-1])])

                # # 连接当前4个角点
                # for coord1, p2 in zip(enumerate(points[:-1]), points[1:]):
                #     idx1, p1 = coord1
                #     self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_num, idx1), 
                #                                         ucommon.gen_node('bound', field_num, (idx1 + 1) % 4), 
                #                                         np.linalg.norm(p1 - p2))])
            
            # 连接3个地块
            self.Graph.add_weighted_edges_from([
                ('bound-0-3', 'bound-1-0', 0.),
                ('bound-0-1', 'bound-2-0', 0.),
                ('bound-0-2', 'bound-2-3', 0.),
                ('bound-1-1', 'bound-2-2', 0.),
            ])

        # sample starts and ends
        self.starts = self.sample_nodes(num_starts) if starts is None else starts
        self.ends = self.sample_nodes(num_ends) if ends is None else ends
        # TODO: change here
        self.num_veh = len(self.starts)
        self.num_endpoints = self.num_veh if self.ends is None else 2*self.num_veh
        self.num_fields = len(self.fields_coords)
        self.utm_base = utm.from_latlon(*center_lat_lon)
        self.crs = "+proj =utm +zone={} +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(self.utm_base[2])
        self.fields = [GeoSeries(geometry.Polygon(points), crs=self.crs) for points in self.fields_coords]
        
        if working_width is None:
            self.working_width = random.sample(self.working_width_range, 1)[0]
        else:
            self.working_width = working_width

        self.fields = [Field(data, 
                             self.working_width, 
                             field_num=idx, 
                             dis_matrix=False, 
                             utm_base=self.utm_base)
                       for idx, data in enumerate(self.fields)]
        
        self.Graph: nx.Graph = nx.compose_all([self.Graph] + [field.Graph for field in self.fields])

        self.make_working_graph()

    def make_working_graph(self, fields: Union[List, str] = 'all'):
        self.working_graph = nx.DiGraph()
        
        [self.working_graph.add_nodes_from([
            (f'start-{idx}', {
            'length': 0,
            'direction_angle': None,
            'anti_direction_angle': None,
            'end0': start,
            'end1': start,
            'embed': torch.Tensor([0, 0, 0])
        })]) for idx, start in enumerate(self.starts)]
        
        if self.ends is not None:
            [self.working_graph.add_nodes_from([(f'end-{idx}', {
                'length': 0,
                'direction_angle': None,
                'anti_direction_angle': None,
                'end0': end,
                'end1': end,
                'embed': torch.Tensor([0, 0, 0])
            })]) for idx, end in enumerate(self.ends)]
        
        self.working_graph: nx.DiGraph = nx.compose_all([self.working_graph] 
                                                        + ([field.working_graph for field in self.fields] if fields == 'all' 
                                                           else [self.fields[i].working_graph for i in fields]))
        
        self.num_nodes = self.working_graph.number_of_nodes()
        self.nodes_list = list(self.working_graph.nodes)
        self.num_working_lines = self.num_nodes - self.num_endpoints
        self.working_line_list = self.nodes_list[len(self.starts)+len(self.ends):]
        self.line_field_idx = [int(name.split('-')[1]) for name in self.working_line_list]
        
        self.whole_D_matrix = np.zeros((self.num_nodes, self.num_nodes, 2, 2))
        self.line_length = []
        for idx1, node1 in enumerate(self.nodes_list):
            self.line_length.append(self.working_graph.nodes[node1]['length'])
            for idx2 in range(max(idx1, self.num_endpoints), self.num_nodes):
                node2 = self.nodes_list[idx2]
                for p1 in [0, 1]: 
                    for p2 in [0, 1]:
                        cartesian_node1 = self.working_graph.nodes[node1][f'end{p1}']
                        cartesian_node2 = self.working_graph.nodes[node2][f'end{p2}']
                        self.whole_D_matrix[idx1, idx2, p1, p2] = self.whole_D_matrix[idx2, idx1, p2, p1] = self.get_dijkstra_path_length(cartesian_node1, cartesian_node2) 
                if idx1 != idx2:
                    if 'end' not in node1 or self.ends is None:
                        self.working_graph.add_edge(node1, node2, 
                                                    dis=self.whole_D_matrix[idx1, idx2], 
                                                    edge_embed=torch.Tensor(self.whole_D_matrix[idx1, idx2].flatten())) # 两个节点之间的边由4个距离组成
                    if 'start' not in node1 or self.ends is None:
                        self.working_graph.add_edge(node2, node1, 
                                                    dis=self.whole_D_matrix[idx2, idx1],
                                                    edge_embed=torch.Tensor(self.whole_D_matrix[idx1, idx2].flatten()))
        self.ori = self.whole_D_matrix[:self.num_veh, self.num_endpoints:] # 各个 start 到各个作业行的距离矩阵
        self.des = self.whole_D_matrix[self.num_veh:self.num_endpoints, self.num_endpoints:] # 各个 end 到各个作业行的距离矩阵
        self.D_matrix = self.whole_D_matrix[self.num_endpoints:, self.num_endpoints:] # 各个作业行之间的距离矩阵
        self.line_length = np.array(self.line_length[self.num_endpoints:])

    def get_fields(self, upper_points, lower_points, rows: int, upper_node=None):
        if rows < 0:
            return
        
        if rows == 0:
            cur_lower_points = lower_points
        else:
            split = 0.5 * np.random.random() + 1 / 2 / (rows + 1)
            line1 = [ucommon.get_point(np.array([upper_points[0], lower_points[0]]), split)]
            split = 0.5 * np.random.random() + 1 / 2 / (rows + 1)
            line1.append(ucommon.get_point(np.array([upper_points[-1], lower_points[-1]]), split))
            line1 = np.array(line1) # 横向切分线

            # 当前横向切分线与所有纵向切分线的交点
            cur_lower_points = [line1[0]] \
                + [ucommon.get_line_cross_point(line1, np.array([p1, p2])) for p1, p2 in zip(upper_points[1:-1], lower_points[1:-1])] \
                + [line1[1]]
        
        cur_upper_node = []
        lower_node = []
        for cur_field_idx, points in enumerate(zip(
                                                    upper_points[:-1],
                                                    cur_lower_points[:-1],
                                                    cur_lower_points[1:],
                                                    upper_points[1:],
                                                    upper_points[:-1]
                                                )):
            field_num = len(self.fields_coords)
            self.fields_coords.append(points)

            # 4角点加入图
            self.Graph.add_nodes_from([(ucommon.gen_node('bound', field_num, idx), {'coord': point}) for idx, point in enumerate(points[:-1])])
            
            # 下方2角点
            lower_node.append(ucommon.gen_node('bound', field_num, 1))
            lower_node.append(ucommon.gen_node('bound', field_num, 2))
            # 上方2角点
            cur_upper_node.append(ucommon.gen_node('bound', field_num, 0))
            cur_upper_node.append(ucommon.gen_node('bound', field_num, 3))

            # # 连接当前4个角点
            # for coord1, p2 in zip(enumerate(points[:-1]), points[1:]):
            #     idx1, p1 = coord1
            #     self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_num, idx1), 
            #                                         ucommon.gen_node('bound', field_num, (idx1 + 1) % 4), 
            #                                         np.linalg.norm(p1 - p2))])
            
            # 连接当前地块和左侧地块
            if cur_field_idx > 0:
                self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_num, 0), 
                                                    ucommon.gen_node('bound', field_num - 1, 3), 
                                                    0)])
                self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_num, 1), 
                                                    ucommon.gen_node('bound', field_num - 1, 2), 
                                                    0)])
        # 连接当前行与上方地块
        if upper_node is not None:
            self.Graph.add_weighted_edges_from([(node1, node2, 0) for node1, node2 in zip(upper_node, cur_upper_node)])

        self.get_fields(cur_lower_points, lower_points, rows - 1, lower_node)

    def render(self, 
               ax = None, 
               working_lines = True, 
               entry_point = False, 
               show = True,
               line_colors = 'blue',
               boundary=False):
        if ax is None:
            _, ax = plt.subplots()
            ax.axis('equal')
        
        if isinstance(line_colors, str):
            line_colors = [line_colors] * self.num_fields
        
        for field, line_color in zip(self.fields, line_colors):
            field.render(ax, working_lines, entry_point, show=False, line_color=line_color, boundary=boundary)
        
        [ax.plot(self.Graph.nodes[start]['coord'][0], 
                self.Graph.nodes[start]['coord'][1], 
                '*y', 
                markersize=20) for start in self.starts]
        if self.ends is not None:
            [ax.plot(self.Graph.nodes[end]['coord'][0], 
                    self.Graph.nodes[end]['coord'][1], 
                    'vr', 
                    markersize=20) for end in self.ends]
        
        ax.set_xlabel('x(m)', fontsize=20)
        ax.set_ylabel('y(m)', fontsize=20)
        if show:
            plt.show()
        return ax
        
    def get_euler_dis(self, node1, node2):
        return ucommon.get_euler_dis(self.Graph, node1, node2)

    def gen_edge(self, node1, node2):
        return ucommon.gen_edge(self.Graph, node1, node2)

    def get_dijkstra_path(self, node1, node2, cartesian=False):
        return ucommon.get_dijkstra_path(self.Graph, node1, node2, cartesian)
    
    def get_dijkstra_path_length(self, node1, node2):
        return ucommon.get_dijkstra_path_length(self.Graph, node1, node2)

    def get_path(self, line1, line2, entry1, entry2, cartesian=False):
        if isinstance(line1, tuple):
            line1 = ucommon.gen_node(*line1)
        if isinstance(line2, tuple):
            line2 = ucommon.gen_node(*line2)
        node1 = self.working_graph.nodes[line1][f'end{entry1}']
        node2 = self.working_graph.nodes[line2][f'end{entry2}']
        path = self.get_dijkstra_path(node1, node2, cartesian)
        if not cartesian:
            return path

        if 'start' not in line1 and 'end' not in line1:
            path[0] = np.append(path[0], self.working_graph.nodes[line1]['direction_angle' if entry1 == 1 else 'anti_direction_angle'])
        if 'start' not in line2 and 'end' not in line2:
            path[-1] = np.append(path[-1], self.working_graph.nodes[line2]['direction_angle' if entry2 == 0 else 'anti_direction_angle'])

        return path

    def sample_nodes(self, num):
        if num == 'all':
            return list(self.Graph.nodes)
        else:
            assert isinstance(num, int), f"input must be integer, got {num}"
            return [random.sample(list(self.Graph.nodes), 1)[0] for _ in range(num)]
    
    def edit_fields(self, delete_lines):
        return 0
        
    
    def merge_field(self, merge_ids : list):
        assert len(merge_ids) > 1, f"input must be looger than 2, got {len(merge_ids)}"
        row, col = self.num_splits[0] + 1, self.num_splits[1] + 1
        print(f'row:{row}, col:{col}')
        
        self.Graph = nx.Graph()
        # check if merge area is a connected component
        def find_row(ids : list, current, connected : list = []):
            ids.pop(ids.index(current))
            connected.append(current)
            if len(ids) == 0:
                return []
            if current + 1 in ids and current + 1 < col:
                find_row(ids, current + 1, connected)
            if current - 1 in ids and current + 1 > 0:
                find_row(ids, current - 1, connected)
            return connected
        
        def find_col(ids : list, current, connected : list = []):
            ids.pop(ids.index(current))
            connected.append(current)
            if len(ids) == 0:
                return []
            if current + col in ids and current + col < row * col:
                find_col(ids, current + col, connected)
            if current - col in ids and current - col > 0:
                find_col(ids, current - col, connected)
            return connected
        if merge_ids == find_row(merge_ids.copy(), merge_ids[0]):
            coords = self.merge_row(merge_ids)
    
        elif merge_ids == find_col(merge_ids.copy(), merge_ids[0]):
            coords = self.merge_col(merge_ids)
        else:
            print('Invalid input!')
        
        self.fields = [GeoSeries(geometry.Polygon(points), crs=self.crs) for points in coords]

        self.fields = [Field(data, 
                             self.working_width, 
                             field_num=idx, 
                             dis_matrix=False, 
                             utm_base=self.utm_base)
                       for idx, data in enumerate(self.fields)]
        
        self.Graph: nx.Graph = nx.compose_all([self.Graph] + [field.Graph for field in self.fields])
        
    def merge_row(self, merge_ids : list):
        row, col = self.num_splits[0] + 1, self.num_splits[1] + 1
        merge_ids.sort()
        fields_coords =  [[point for point in fields_coord] for fields_coord in self.fields_coords]
        
        self.num_fields = len(self.fields_coords) - len(merge_ids) + 1
        left, right = merge_ids[0], merge_ids[-1]
        mrow, mcol_left, mcol_right = left % col, left % row, right % row
        fields_coords[left][3] = fields_coords[right][3]
        fields_coords[left][2] = fields_coords[right][2]
        for r in range(row):
            for c in range(col):
                field_idx = r * col + c
                fields_coord = fields_coords[field_idx]
                if field_idx <= left:
                    self.Graph.add_nodes_from([(ucommon.gen_node('bound', field_idx, idx), {'coord': point}) for idx, point in enumerate(fields_coord[:-1])])
                    if c > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                            ucommon.gen_node('bound', field_idx - 1, 3), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 1), 
                                                            ucommon.gen_node('bound', field_idx - 1, 2), 
                                                            0)])
                    if r > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                            ucommon.gen_node('bound', field_idx - col, 1), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 3), 
                                                            ucommon.gen_node('bound', field_idx - col, 2), 
                                                            0)])
                elif field_idx > right:
                    field_idx -= len(merge_ids) - 1
                    self.Graph.add_nodes_from([(ucommon.gen_node('bound', field_idx, idx), {'coord': point}) for idx, point in enumerate(fields_coord[:-1])])
                    if c > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                            ucommon.gen_node('bound', field_idx - 1, 3), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 1), 
                                                            ucommon.gen_node('bound', field_idx - 1, 2), 
                                                            0)])
                    if r > 0:
                        if field_idx > left + col:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                                ucommon.gen_node('bound', field_idx - col, 1), 
                                                                0)])
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 3), 
                                                                ucommon.gen_node('bound', field_idx - col, 2), 
                                                                0)])
                        elif field_idx < left + col - len(merge_ids) + 1:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                                ucommon.gen_node('bound', field_idx - col + len(merge_ids) - 1, 1), 
                                                                0)])
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 3), 
                                                                ucommon.gen_node('bound', field_idx - col + len(merge_ids) - 1, 2), 
                                                                0)])
                        elif c == mcol_left:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 0), 
                                                                ucommon.gen_node('bound', field_idx - col + len(merge_ids) - 1, 1), 
                                                                0)])
                        elif c == mcol_right:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', field_idx, 3), 
                                                                ucommon.gen_node('bound', field_idx - col, 2), 
                                                                0)])
        coords = []
        for idx, fields_coord in enumerate(fields_coords):
            if idx not in merge_ids[1:]:
                coords.append(fields_coord)
        return coords
    
    def merge_col(self, merge_ids : list):
        row, col = self.num_splits[0] + 1, self.num_splits[1] + 1
        merge_ids.sort()
        fields_coords =  [[point for point in fields_coord] for fields_coord in self.fields_coords]  
        def transpose(idx):
            r = idx // row
            c = idx % row
            return c * row + r
        def inv_transpose(idx):
            r = idx % col
            c = idx // col
            return r * col + c
        self.num_fields = len(self.fields_coords) - len(merge_ids) + 1
        up, down = merge_ids[0], merge_ids[-1]
        left, right = transpose(up), transpose(down)
        mrow, mcol_left, mcol_right = left % col, left % row, right % row
        fields_coords[inv_transpose(left)][1] = fields_coords[inv_transpose(right)][1]
        fields_coords[inv_transpose(left)][2] = fields_coords[inv_transpose(right)][2]
        for r in range(row):
            for c in range(col):
                
                field_idx = r * col + c
                fields_coord = fields_coords[inv_transpose(field_idx)]
                if field_idx <= left:
                    self.Graph.add_nodes_from([(ucommon.gen_node('bound', inv_transpose(field_idx), idx), {'coord': point}) for idx, point in enumerate(fields_coord[:-1])])
                    if c > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - 1), inv_transpose(3)), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(1)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - 1), inv_transpose(2)), 
                                                            0)])
                    if r > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - col), inv_transpose(1)), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(3)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - col), inv_transpose(2)), 
                                                            0)])
                elif field_idx > right:
                    field_idx -= len(merge_ids) - 1
                    self.Graph.add_nodes_from([(ucommon.gen_node('bound', inv_transpose(field_idx), idx), {'coord': point}) for idx, point in enumerate(fields_coord[:-1])])
                    if c > 0:
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - 1), inv_transpose(3)), 
                                                            0)])
                        self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(1)), 
                                                            ucommon.gen_node('bound', inv_transpose(field_idx - 1), inv_transpose(2)), 
                                                            0)])
                    if r > 0:
                        if field_idx > left + col:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col), inv_transpose(1)), 
                                                                0)])
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(3)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col), inv_transpose(2)), 
                                                                0)])
                        elif field_idx < left + col - len(merge_ids) + 1:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col + len(merge_ids) - 1), inv_transpose(1)), 
                                                                0)])
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(3)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col + len(merge_ids) - 1), inv_transpose(2)), 
                                                                0)])
                        elif c == mcol_left:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(0)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col + len(merge_ids) - 1), inv_transpose(1)), 
                                                                0)])
                        elif c == mcol_right:
                            self.Graph.add_weighted_edges_from([(ucommon.gen_node('bound', inv_transpose(field_idx), inv_transpose(3)), 
                                                                ucommon.gen_node('bound', inv_transpose(field_idx - col), inv_transpose(2)), 
                                                                0)])
        coords = []
        for idx, fields_coord in enumerate(fields_coords):
            if idx not in merge_ids[1:]:
                coords.append(fields_coord)
        return coords
    
    def from_ia(self, ia):
        self.__dict__.update(ia.__dict__)
        del self.home
        self.fields = [Field(data, 
                                self.working_width, 
                                field_num=idx, 
                                dis_matrix=False, 
                                utm_base=self.utm_base)
                        for idx, data in enumerate([GeoSeries(geometry.Polygon(points), crs=self.crs) 
                                                    for points in self.fields_coords])]
        self.make_working_graph()

if __name__ == '__main__':
    from datetime import datetime
    # multi = multiField([1, 1], 'T', working_width=6) 
    multi = multiField([1, 1], working_width=1.8, center_lat_lon=(23.243233258, 113.637738798), 
                        boundary_coords=np.array([[0.01, 0.01],
                                                  [92.56700655078748,-35.4055412709713],
                                                  [85.92209068464581,-69.8845205558464],
                                                  [-15.824082368111704,-30.653775474522263]]),
                        num_starts=2, num_ends=1)
    #    print(multi.boundary)
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(multi.boundary_coords[:, 0], multi.boundary_coords[:, 1], 'k', linewidth=2)
    ax1.axis('equal')
    multi.render(ax1, entry_point=True, show=False)

    # print(multi.get_dijkstra_path(('line_0', 1, 2), ('line_b_1', 2, 2)))
    # dijk_path = multi.get_dijkstra_path(('line_0', 1, 2), ('line_b_1', 2, 2), cartesian=True)
    # dijk_len = multi.get_dijkstra_path_length(('line_0', 1, 2), ('line_b_1', 2, 2))
    # path_g = dubins.Dubins(3, 0.5)
    # path, info = path_g.dubins_multi(dijk_path)
    # # path = np.array(path)
    
    # dijk_path = np.array(dijk_path)

    # plt.plot(dijk_path[:, 0], dijk_path[:, 1], 'r', linewidth=3)
    # plt.plot(dijk_path[:, 0], dijk_path[:, 1], 'go', markersize=5)
    
    # plt.plot(path[:, 0], path[:, 1], 'b--', linewidth=2)

    # node_path = multi.get_path('line-1-1', 'start', 1, 1, True)
    # path = path_g.dubins_multi(node_path)[0]
    # plt.plot(path[:, 0], path[:, 1], 'k--', linewidth=2)
    # plt.show()
    
    # multi.merge_field([0, 4, 8])
    # ax2 = plt.subplot(2, 1, 2)
    # ax2.plot(multi.boundary_coords[:, 0], multi.boundary_coords[:, 1], 'k', linewidth=2)
    # ax2.axis('equal')
    # multi.render(ax2, entry_point=True, show=False)
    
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d_%H-%M")
    save_dir = f'/home/xuht/Intelligent-Agriculture/mdvrp/env/test_{formatted_date}.png'
    plt.savefig(save_dir)
    print(f'Fig saved in {save_dir}')
    
