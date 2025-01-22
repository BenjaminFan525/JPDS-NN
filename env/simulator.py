import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image
import env.dubins as dubins
from env.car import car, Robot, DRIVE_MODE
from env.multi_field import multiField
import utils.common as ucommon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2
from typing import Union, List, Optional

COLORS = (
    [
        # deepmind style
        'red',
        'yellow',
        'black',
        'green',
        'purple',
        'blue',
        'orange',
        # '#0072B2',
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

# class simCar:
#     def __init__(self, car, path) -> None:
#         self.car = car
#         self.path = path
#         self.tracking = False
#         self.t = None

# class Path:
#     def __init__(self, path: np.ndarray, working_list: List[bool],
#                  straight: List[bool], dijkstra_path: List[str]) -> None:
#         self.path = path
#         self.working_list = working_list
#         self.straight = straight
#         self.dijkstra_path = dijkstra_path
#         self.length = len(self.path)

#     def __len__(self):
#         return self.length

class basicSimulator:
    def __init__(self, min_R = 0, path_sep = None):
        self.min_R = min_R
        if path_sep:
            self.path_sep = path_sep
        else:
            self.path_sep = self.min_R / 2
        self.path_g = dubins.Dubins(self.min_R, self.path_sep)
        self.tracking_vector = None
        self.working = False

    def get_pose(self):
        raise NotImplementedError
    
    def set_goal(self, targets, working_list=None):
        self.path, info = self.path_g.dubins_multi(targets)
        
        self.straight = info['straight']
        self.working_list = [False] * len(self.path)
        if working_list:
            for idx in range(len(working_list) - 1):
                # print(self.working_list)
                self.working_list[info['sp'][idx]:info['sp'][idx + 1]] = [working_list[idx]] * (info['sp'][idx + 1] - info['sp'][idx])
        
        self.tracking_vector = None

    def start(self):
        if len(self.path) < 2:
            self.stop()
        else:
            t = self.path[1] - self.path[0]
            t = t / np.linalg.norm(t)
            self.tracking_vector = t
            self.working = self.working_list[0]
            # self.physic_step()

    def physic_step(self):
        '''
        take a step in simulation or send a command in real world experiment
        '''
        raise NotImplementedError
    
    def stop(self):
        self.tracking_vector = None
        self.working = False
        self.working_list = []
        self.straight = []
        self.path = np.array([])

    def step(self, car_pose: ndarray, on_change_func = None, args = {}):
        # if self.render:
        #     self.ax.plot(car_pose[0], car_pose[1], 'og')
        #     plt.pause(0.1)
        if self.tracking_vector is None:
            self.stop()
            return
        # 经过了终点，换下一个路径
        if (car_pose - self.path[1]) @ self.tracking_vector >= 0:
            self.path = np.delete(self.path, 0, axis=0) # 删除路径中的第一个点(原路径起点)
            self.straight.pop(0)
            self.working_list.pop(0)
            self.working = self.working_list[0]
            if on_change_func is not None:
                on_change_func(**args)

            if len(self.straight) < 2:
                self.stop()
                return
            else:
                t = self.path[1] - self.path[0]
                self.tracking_vector = t / np.linalg.norm(t)

        self.physic_step()
    
    def on_working(self):
        self.working = True
    
    def off_working(self):
        self.working = False

class Simulator:
    def __init__(self, path: list, car_list: list, line_nodes: list, lines_id: list, field: multiField, 
                 working_list: list = None, car_tracking = None, ctrl_period = 0.1, max_step = 5000,
                 time_teriminate=None) -> None:
        self.path_list = path # 路径列表
        self.car_list = car_list # 车辆列表
        self.working_list = working_list if working_list is not None else [[True for _ in range(len(p))] for p in self.path_list]
        self.line_nodes = line_nodes
        self.lines_id = lines_id
        self.field = field
        self.time_teriminate = time_teriminate
        # self.car_status = ['home', [0, 0]]*len(car_list)
        self.car_status = [{'line': self.field.nodes_list[0+idx], 'entry': None, 'pos': path[0][0], 'inline': False, 'teriminate': False, 'traveled': []} for idx, _ in enumerate(range(len(car_list)))]
        self.traveled_line = [[] for _ in range(len(car_list))]
        if car_tracking:
            self.car_tracking = car_tracking # 车辆是否跟踪路径
        else:
            self.car_tracking = [True] * len(car_list)
        for car, tracking, working in zip(self.car_list, self.car_tracking, self.working_list):
            if tracking and hasattr(car, 'working'):
                car.working = working[0]
        self.tracking_vector_list = []
        for path in self.path_list:
            if len(path) < 2:
                self.tracking_vector_list.append(None)
            else:
                while np.allclose(path[0], path[1]):
                    path = path[1:, :]
                t = path[1] - path[0]
                t = t / np.linalg.norm(t)
                self.tracking_vector_list.append(t)

        self.drive_mode = "physical"
        self.drive_mode_ = DRIVE_MODE.AUTO_VEL_PSI

        self.ctrl_period = ctrl_period # 上位机控制周期
        self.inner_period = self.car_list[0].ctrl_period # 内环PID控制周期
        self.inner_loop_times = int(self.ctrl_period / self.inner_period) # 每次上位机控制后的内环PID控制次数
        self.ctrl_period = self.inner_loop_times * self.inner_period

        self.time = 0.
        self.steps: int = 0
        self.max_step = max_step
        self.terminate = False
        # self.max_step = len(self.path_list[0])//2
    
    # One contrl step. 
    # 'a' means 'action'. 
    # Action should be a ndarray with shape (number of cars)x2 or 2x(number of cars). 
    # An action of a car is a tuple (v, psi), which defines the desired velosity vector in the car's body axis. 
    # If a car is in tracking mode, the action tuple represents (working velosity, driving velosity). 
    # Return True if all cars in tracking mode finished their job and the other cars have stopped.
    # Else return False.
    def step(self, a: np.ndarray):
        
        # assert a.shape == (len(self.car_list), 2) or a.shape == (2, len(self.car_list))
        # if a.shape == (2, len(self.car_list)):
        #     a = a.T

        self.terminate = True
        for idx, car in enumerate(self.car_list):
            if self.car_tracking[idx]:
                # 没有要跟踪的路径， 停车
                if self.tracking_vector_list[idx] is None:
                    car.stop()
                # 经过了终点，换下一个路径
                elif (car.position() - self.path_list[idx][1]) @ self.tracking_vector_list[idx] >= 0:
                    if len(self.line_nodes[idx]) and np.allclose(self.path_list[idx][0], self.line_nodes[idx][0]):
                        self.line_nodes[idx] = np.delete(self.line_nodes[idx], 0, axis=0)
                        if len(self.line_nodes[idx]) % 2 == 0:
                            line, entry = self.lines_id[idx].pop(0)
                            self.traveled_line[idx].append([line, entry])
                            self.car_status[idx]['traveled'].append(self.field.working_line_list[line])
                    
                    self.car_status[idx]['line'] = self.field.working_line_list[self.lines_id[idx][0][0]] if len(self.lines_id[idx]) else self.field.nodes_list[idx+len(self.car_list)]
                    self.car_status[idx]['entry'] = self.lines_id[idx][0][1] if len(self.lines_id[idx]) else None
                    self.car_status[idx]['pos'] = self.path_list[idx][0]
                    self.car_status[idx]['inline'] = len(self.line_nodes[idx]) % 2 != 0
                    
                    self.path_list[idx] = np.delete(self.path_list[idx], 0, axis=0) # 删除路径中的第一个点(原路径起点)
                    self.working_list[idx].pop(0)
                    if len(self.path_list[idx]) < 2:
                        self.tracking_vector_list[idx] = None
                        car.stop()
                    elif self.time_teriminate and self.time > self.time_teriminate:
                        self.car_status[idx]['teriminate'] = True
                        car.stop()
                    else:
                        self.terminate = False
                        self.car_status[idx]['teriminate'] = False
                        if hasattr(car, 'working'):
                            car.working = self.working_list[idx][0]
                        t = self.path_list[idx][1] - self.path_list[idx][0]
                        self.tracking_vector_list[idx] = t / np.linalg.norm(t)
                
                if self.tracking_vector_list[idx] is not None and self.car_status[idx]['teriminate'] == False:
                    self.terminate = False
                    car.follow_trail(self.tracking_vector_list[idx], 
                                     self.path_list[idx][0], 
                                     a[idx][0] if self.working_list[idx][0] else a[idx][1], 
                                     follow = False)
            else:
                car.v_des, car.psi_des = a[idx]
            
            for _ in range(self.inner_loop_times):
                car.update_state(self.drive_mode_)
                # if idx == 0:
                #     print(car.state.v)

            if car.v() > 0:
                self.terminate = False

        self.steps += 1
        self.time += self.ctrl_period

        if self.steps >= self.max_step:
            self.terminate = True
        return self.terminate 

    # Finish one trajectory
    def rollout(self, a, render = False, ax = None, update = True, show = True, output_figure = False, label=False):
        # ori_ax = deepcopy(ax)
        
        if render:
            if ax is None:
                _, ax = plt.subplots()
                ax.axis('equal')
            ori_ax = len(ax.lines)
            if output_figure:
                figures = []
            text = ax.text(0.01, 0.99, 'Initializing', 
                        horizontalalignment='left', 
                        verticalalignment='top',
                        transform=ax.transAxes)
        
        while not self.step(a):
            if render:
                if self.steps % 50 == 0:
                    if update:
                        while len(ax.lines) > ori_ax:
                            ax.lines.pop()
                    self.render(ax, show = False, label=label)
                    # disp_str = ''.join([f"Car {idx+1} | line {status['line']} | pos ({status['pos'][0]:.2f}, {status['pos'][1]:.2f}) | inline {status['inline']}\n" for idx, status in enumerate(self.car_status)])
                    disp_str = str(self.time)
                    text.set_text(disp_str)
                    if output_figure:
                        canvas = FigureCanvasAgg(plt.gcf())
                        w, h = canvas.get_width_height()
                        canvas.draw()
                        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                        buf.shape = (w, h, 3)
                        buf = buf[:, :, [2, 1, 0]]
                        # buf = np.roll(buf, 3, axis=2)
                        image = Image.frombytes("RGB", (w, h), buf.tobytes())
                        figures.append(np.asarray(image)[:, :, :3])
                    if show:
                        plt.pause(0.01)

        return figures if render and output_figure else None
            
    # def pause(self):
    #     delete_lines = []
    #     new_homes = []
    #     for car, line in zip(self.car_status, self.traveled_line):
    #         delete_lines += line
    #         # new_homes += car['line']
    #         if car['inline']:
    #             delete_lines += car['line']
        
        
            
            
    
    def render(self, ax = None, show = True, label=False):
        if ax is None:
            _, ax = plt.subplots()
            ax.axis('equal')

        for idx, car in enumerate(self.car_list):
            if label:
                car.plot(color = COLORS[idx % len(COLORS)], mode = 1, ax = ax, label=f'Veh{idx+1}')
            else:
                car.plot(color = COLORS[idx % len(COLORS)], mode = 1, ax = ax)
        ax.legend(fontsize=16, loc='lower left')

        # print(self.working_direction)   
        if show:
            plt.show()
            
        return ax 
    
    def set_drive_mode(self, mode):
        if mode == "physical":
            self.drive_mode_ = DRIVE_MODE.AUTO_VEL_PSI
        else:
            assert mode == "direct"
            self.drive_mode_ = DRIVE_MODE.DIRECT_PSI
        self.drive_mode = mode

    # 设置车辆工作模式 
    # tracking=True表示跟踪路径，action=(v, _)
    # tracking=False表示直接控制，action=(v, psi)
    def set_tracking_mode(self, idx, tracking):
        if isinstance(idx, int):
            self.car_tracking[idx] = tracking
        else:
            for i in idx:
                self.car_tracking[i] = tracking[i]

    def set_path(self, idx, path):
        if isinstance(idx, int):
            self.path_list[idx] = path
        else:
            for i in idx:
                self.path_list[i] = path[i]
    
    def add_path(self, idx, path):
        if isinstance(idx, int):
            np.vstack(self.path_list[idx], path)
        else:
            for i in idx:
                np.vstack(self.path_list[i], path[i])
    
    def pop(self, idx):
        if isinstance(idx, int):
            self.path_list.pop(idx)
            self.car_list.pop(idx)
        else:
            idx.sort()
            for i in idx[::-1]:
                self.path_list.pop(i)
                self.car_list.pop(i)
    
    def add(self, path, car):
        if isinstance(path, list):
            assert len(path) == len(car)
            self.path_list += path
            self.car_list += car
            for path in self.path_list:
                if len(path) < 2:
                    self.tracking_vector_list.append(None)
                else:
                    t = path[1] - path[0]
                    t = t / np.linalg.norm(t)
                    self.tracking_vector_list.append(t)
        else:
            self.path_list.append(path)
            self.car_list.append(car)
            if len(path) < 2:
                self.tracking_vector_list.append(None)
            else:
                t = path[1] - path[0]
                t = t / np.linalg.norm(t)
                self.tracking_vector_list.append(t)
            

def get_working_path(path_g, targets, ori_working_list):
    if isinstance(path_g, dubins.Dubins):
        path_g = [path_g] * len(targets)
    paths = []
    working_lists = []
    info_list = []
    for target, working_list, generator in zip(targets, ori_working_list, path_g):
        if len(target) > 1:
            path, info = generator.dubins_multi(target)
        else:
            path = np.array(target)
            info = {'straight': [0], 'sp':[0], 'length': 0.}
        paths.append(path)
        info_list.append(info)
        working_ = []
        for idx, working in enumerate(working_list[:-1]):
            # print(self.working_list)
            working_ += [working for _ in range(info['sp'][idx + 1] - info['sp'][idx])]
        if len(working_) == 0:
            working_ = [False]
        working_ += [working_[-1]]
        working_lists.append(working_[1:])
    return paths, working_lists, info_list

class arrangeSimulator():
    def __init__(self, field: multiField, car_cfg) -> None:
        self.field = field
        self.car_cfg = car_cfg
        
    def init_simulation(self, arrangements, debug = False, drive_mode = 'direct', init_simulator = True, dense_straight = True, path_split = None):
        targets = []
        ori_working_list = []
        lines_id = [[] for _ in arrangements]
        line_nodes = [[] for _ in arrangements]
        for idx, arrange in enumerate(arrangements):
            if len(arrange) == 0:
                if np.allclose(self.field.Graph.nodes[self.field.starts[idx]]['coord'], self.field.Graph.nodes[self.field.ends[idx]]['coord']):
                    targets.append([self.field.Graph.nodes[self.field.starts[idx]]['coord']])
                    ori_working_list.append([False])
                else:
                    targets.append(self.field.get_path(f'start-{idx}', f'end-{idx}', 1, 1, True))
                    ori_working_list.append([False, False])
                continue
            lines_id[idx] = [line for line in arrange]
            targets.append(self.field.get_path(f'start-{idx}', self.field.working_line_list[arrange[0][0]], 1, arrange[0][1], True))
            # if len(targets[idx]) > 1:
            ori_working_list.append([False, True])
            # else:
            #     ori_working_list.append([False])
            for working_line in arrange:
                line_name = self.field.working_line_list[working_line[0]]
                point_name = [
                    self.field.working_graph.nodes[line_name][f'end{working_line[1]}'],
                    self.field.working_graph.nodes[line_name][f'end{1-working_line[1]}']
                ]
                line_nodes[idx] += [self.field.Graph.nodes[name]['coord'] for name in point_name]
            
            for line1, line2 in zip(arrange[:-1], arrange[1:]):
                delta_target = self.field.get_path(self.field.working_line_list[line1[0]], self.field.working_line_list[line2[0]], 1 - line1[1], line2[1], True)
                targets[idx] += delta_target
                ori_working_list[idx] += [False, True]
            if self.field.ends is None:
                delta_target = self.field.get_path(self.field.working_line_list[arrange[-1][0]], f'start-{idx}', 1 - arrange[-1][1], 1, True)
            else:
                delta_target = self.field.get_path(self.field.working_line_list[arrange[-1][0]], f'end-{idx}', 1 - arrange[-1][1], 1, True)
            targets[idx] += delta_target
            ori_working_list[idx] += [False, False]      
        
        path_g = [dubins.Dubins(cfg['min_R'], 
                                np.clip(cfg['min_R'] / 4, 1, 5) if path_split is None else path_split, 
                                dense_straight) 
                    for cfg in self.car_cfg]
        self.paths, working_list, info_list = get_working_path(path_g, targets, ori_working_list)
        
        if not init_simulator:
            return targets, ori_working_list
        
        first_dir = [0 for _ in range(len(self.car_cfg))]
        for idx, path in enumerate(self.paths):
            if len(path) < 2:
                continue
            while np.allclose(path[0], path[1]):
                path = path[1:, :]
            first_dir_tmp = path[1] - path[0]
            first_dir[idx] = np.arctan2(first_dir_tmp[1], first_dir_tmp[0])

        cars = [Robot({'car_model': "car3", 'working_width': self.field.working_width, **cfg}, 
                 state_ini=[path[0, 0], path[0, 1], 0, 0, f_dir, 0], debug=debug) 
            for path, f_dir, cfg in zip(self.paths, first_dir, self.car_cfg)]
            
        self.simulator = Simulator(self.paths, cars, line_nodes, lines_id, self.field, working_list, max_step=1000000)
        self.simulator.set_drive_mode(drive_mode)
        return self.paths, working_list, info_list
        
    def render_arrange(self, ax=None):
        if ax is None:
            ax = self.field.render(working_lines=False, show=False)
        else:
            self.field.render(ax, working_lines=False, show=False)
        for idx, path in enumerate(self.paths):
            ax.plot(path[:, 0], path[:, 1], '--', color = COLORS[idx % len(COLORS)], label=f'Veh{idx+1}')
        ax.legend(fontsize=16, loc='lower left')
        return ax
    
    def simulate(self, ax = None, render = False, show = True, output_figure = False, label = False):
        if ax == None and render:
            ax = self.render_arrange()
        
        velosity = np.array([[cfg['vw'], cfg['vv']] for cfg in self.car_cfg])
        figs = self.simulator.rollout(velosity, render=render, ax=ax, show=show, output_figure=output_figure,label=label)
        s = 0
        c = 0
        car_time = []
        car_dis = []
        for robo in self.simulator.car_list:
            s += robo.total_distance
            c += robo.total_cost
            if robo.debug:
                car_time.append([robo.tv, robo.tw])
                car_dis.append([robo.driving_distance, robo.working_distance])
            else:
                car_time.append(None)
                car_dis.append(None)
        t = self.simulator.time
        return t, c, s, car_time, car_dis, ax, figs

if __name__ == "__main__":
    import env.dubins as dubins
    from shapely import geometry
    import geopandas
    import utm
    from geopandas import GeoSeries
    import pandas as pd

    from env.field import Field
    from env.car import car, Robot
    from env.multi_field import multiField
    from algo import MGGA
    from utils import fit, load_dict, decode
    import os
    from model.ac import load_model
    from copy import deepcopy
    from torch_geometric.data import Batch
    import torch
    from torch_geometric.utils import unbatch, from_networkx
    from datetime import datetime
    from utils import save_dict

    # dir = input("输入作业地块: ")
    # if dir.split('.')[-1] == 'geojson':
    #     data = geopandas.read_file(dir)
    # else:
    #     data = dir.split(';')
    #     points = []
    #     for point in data:
    #         points.append((
    #             float(point.split(',')[0]),
    #             float(point.split(',')[1])
    #         ))
    #     points.append(points[-1])
    #     data = GeoSeries(geometry.Polygon(points), crs='EPSG:4326')
    


    # working_width = float(input("输入作业行间距："))
    # # field = Field(data, working_width)
    # field = multiField([1, 1], width = (1000, 1000), working_width=working_width) 
    # field = load_dict("/home/xuht/Intelligent-Agriculture/Dataset/Gdataset/Task_Validation/2_3/0_35_24/field.pkl")


    # car_cfg = [
    #     {'vv': 7, 'tt': 0, 'cw': 0.008, 'cv': 0.003, 'vw': 2, 'min_R': 6},
    #     {'vv': 7, 'tt': 0, 'cw': 0.008, 'cv': 0.004, 'vw': 2, 'min_R': 6},
    #     {'vv': 7, 'tt': 0, 'cw': 0.01, 'cv': 0.006, 'vw': 4, 'min_R': 6},
    # ]
    data = "/home/xuht/Intelligent-Agriculture/MAdata/4_4/0_39_24"
    # data = "/home/xuht/Intelligent-Agriculture/MAdata/3_4/0_39_12"
    field_ia = load_dict(os.path.join(data, 'field.pkl'))
    car_cfg = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 6},
               {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 6},]
    car_cfg_0 = [{'vw': 2.5, 'vv': 4.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
               {'vw': 3, 'vv': 5, 'cw': 0.007, 'cv': 0.005, 'tt': 0, 'min_R': 0.0001},
               {'vw': 3, 'vv': 5.5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
               {'vw': 2, 'vv': 5, 'cw': 0.008, 'cv': 0.006, 'tt': 0, 'min_R': 0.0001},
               {'vw': 3, 'vv': 6, 'cw': 0.01, 'cv': 0.008, 'tt': 0, 'min_R': 0.0001},]
    algo = 'PPO-c'
    f = 't'
    GA = MGGA(f = f, order = True, gen_size=200, max_iter=200)
    # tS = np.zeros((len(car_cfg), len(field.line_length)))
    # for i in range(len(car_cfg)):
    #     tS[i] = field.line_length / car_cfg[i]['vw']

    # GA = MGGA(f = 't', order = True, render = False, gen_size = 200, max_iter = 200)
    # T, best, log = GA.optimize(field.D_matrix, 
    #                            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #                            car_cfg, 
    #                            field.line_length, 
    #                            np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #                            field.line_field_idx)
    
    # checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__21-43_s/best_model14.pt"
    # checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-05__11-05_c/best_model23.pt"
    checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt"
    # checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_safe_rl/esb_ppo_lag/2024-03-14__14-35_L15/best_model21.pt"
    # checkpoint = "/home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_safe_rl/esb_ppo_lag/2024-03-14__14-37_L20/best_model24.pt"
    car_cfg_v = [[cur_car['vw'], cur_car['vv'],
                    cur_car['cw'], cur_car['cv'],
                    cur_car['tt']] for cur_car in car_cfg]
    car_tensor = torch.tensor(np.array(car_cfg_v)).float()
    
    field = multiField(num_splits=[1, 1], 
                       starts=['bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1', 
                               'bound-2-1'],
                       ends=['bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1',
                             'bound-2-1']
                       )
    
    # field = multiField(num_splits=[1, 1], 
    #                    starts=['bound-2-1', 
    #                           'bound-1-3', 
    #                           'bound-3-2', 
    #                           'bound-0-0', 
    #                           'bound-0-0'])
    # field = multiField([1, 1], homes=['bound-2-1'])
    
    field.from_ia(field_ia)
    field_list = [0, 3]
    field.make_working_graph(field_list)
    save_dict(field, '/data/fanyx_files/Intelligent-Agriculture/MAdata/4_4/0_39_24/field')
    
    ac = load_model(checkpoint)
    ac.num_veh = field.num_veh
    ac.sequential_sel = True
    ac.end_en = True if field.ends is not None else False
    ac.encoder.end_en = ac.end_en
    
    pygdata = from_networkx(field.working_graph, group_node_attrs=['embed'], group_edge_attrs=['edge_embed'])
    # pygdata = load_dict(os.path.join(data, 'pygdata.pkl'))
    data_t = {'graph': Batch.from_data_list([pygdata]), 'vehicles': car_tensor.unsqueeze(0)}
    info = {'veh_key_padding_mask': torch.zeros((1, len(car_tensor))).bool(), 'num_veh': torch.tensor([[car_tensor.shape[0]]])}

    with torch.no_grad():
        seq_enc, _, _, _, _, _, _ = ac(data_t, info, deterministic=True, criticize=False, )
    
    T = decode(seq_enc[0])
    
    figure, ax = plt.subplots()
    ax.axis('equal')
    name = 'dijkstra_path.png'
    path = os.path.join(data, name)
    sim = arrangeSimulator(field, car_cfg_0)
    sim.init_simulation(T, init_simulator=False)
    sim.render_arrange(ax)
    plt.savefig(path)

    print("Initializing")
    simulator = arrangeSimulator(field, car_cfg)
    simulator.init_simulation(T, debug=True)
    simulator.render_arrange()
    name = 'dubins_path.png'
    path = os.path.join(data, name)
    plt.savefig(path)
    # cars = [Robot({'car_model': "car3", 'working_width': working_width, **cfg}, 
    #              state_ini=[path[0, 0], path[0, 1], 0, 0, f_dir, 0], debug=True) 
    #         for path, f_dir, cfg in zip(paths, first_dir, car_cfg)]
    # simulator = Simulator(paths, cars, working_list, max_step=500000)
    # simulator.set_drive_mode("direct")#physical
    print("Start simulation")
    t, c, s, car_time, car_dis, figs = simulator.simulate(True, True, True)
    # plt.show()
    name = 'working_result.png'
    path = os.path.join(data, name)
    plt.savefig(path)
    print(path)
    
    # chosen_idx = [[line[0] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]
    # chosen_entry = [[line[1] for line in simulator.simulator.traveled_line[idx]] for idx in range(len(car_cfg))]

    if figs is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(data, algo + ".mp4"), fourcc, 12, (figs[0].shape[1], figs[0].shape[0]), True)
        # map(videoWriter.write, figs)
        for fig in figs:
            videoWriter.write(fig)
        videoWriter.release() 
    else:
        print('error')
        
    print(os.path.join(data, algo + ".mp4"))
    
    # simulator.rollout(np.array([[2, 2], [2, 2], [4, 4]]), render=True, ax=ax)#
    # s = 0
    # c = 0
    # car_time = []
    # car_dis = []
    # for robo in simulator.car_list:
    #     s += robo.total_distance
    #     c += robo.total_cost
    #     car_time.append([robo.tv, robo.tw])
    #     car_dis.append([robo.driving_distance, robo.working_distance])
    # t = simulator.time
    print(car_time)
    print(car_dis)
    print(np.sum(field.line_length))
    # t_exp = 1 / fit(field.D_matrix, 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
    #             car_cfg, 
    #             field.line_length,
    #             T,
    #             type = 't',
    #             tS_t=False)
    # s_exp = 1 / fit(field.D_matrix, 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
    #             car_cfg, 
    #             tS,
    #             T,
    #             type = 's') + np.sum(field.line_length)
    # c_exp = 1 / fit(field.D_matrix, 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)), 
    #             np.tile(field.ori, (len(car_cfg), 1, 1, 1)),
    #             car_cfg, 
    #             tS,
    #             T,
    #             type = 'c')
    
    # print(f'Time expected: {np.round(t_exp, 2)} s, real time: {np.round(t, 2)} s. ')
    # print(f'Distance expected: {np.round(s_exp, 2)} m, real distance: {np.round(s, 2)} m. ')
    # print(f'Energy cost expected: {np.round(c_exp, 2)} L, real energy cost: {np.round(c, 2)} L. ')
    # print(f'Max e_n: {np.round(np.max([robo.total_ect for robo in simulator.simulator.car_list])/simulator.simulator.steps, 2)} m. ')
    # print(f'Max e_t: {np.round(np.max([robo.total_ev for robo in simulator.simulator.car_list])/simulator.simulator.steps, 2)} m/s. ')
    # plt.show()
    