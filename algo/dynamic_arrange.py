import numpy as np
from enum import Enum

S_state = Enum("S_state",('Available','Unavailable','Occupied'))

class IIG:
    _defaults = {
        "initialize"    : True,

        "enable_log"    : True,

        "zero_base"     : False,

    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, D, ori, car_cfg, S_cfg, width, des = None, static_arrange = None, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.D = D
        self.car_cfg = car_cfg
        self.S_cfg = S_cfg
        self.des = des
        min_r = [cfg['min_R'] for cfg in car_cfg]
        min_r = np.array(min_r)
        self.width = width
        self.min_margin = ((2 * min_r / width).astype(int) + 1).tolist()
        self.W = (-np.ones(len(car_cfg))).tolist()

        if self.enable_log:
            self.log = []
            for _ in range(len(self.car_cfg)):
                self.log.append([])

        self.O = []
        for _ in range(len(S_cfg)):
            self.O.append(S_state.Available)

        if self.initialize:
            w = [cfg['w'] for cfg in car_cfg]
            order = np.argsort(w)[::-1].tolist()  # sort the cars in the w order
            for i in order:
                S_order = np.argsort(ori[i])
                for j in S_order:
                    if self.O[j].value == S_state.Available.value:
                        self.W[i] = j
                        if self.enable_log:
                            if self.zero_base:
                                self.log[i].append(j)
                            else:
                                self.log[i].append(j + 1)
                        self.set_S(j, S_state.Occupied)
                        break
        else:
            self.W = static_arrange
            for i, j in enumerate(self.W):
                self.set_S(j, S_state.Occupied) 
                if self.enable_log:
                    if self.zero_base:
                        self.log[i].append(j)
                    else:
                        self.log[i].append(j + 1)

        self.complete = False

    def is_complete(self):
        return self.complete

    def set_S(self, id, state):
        self.O[id] = state

    def set_car_S(self, id, state):
        self.O[self.W[id]] = state

    def add_car(self, car_cfg, ori):
        self.car_cfg.append(car_cfg)
        self.min_margin.append(int(2 * car_cfg['min_R'] / self.width) + 1)
        S_order = np.argsort(ori)
        for j in S_order:
            if self.O[j].value == S_state.Available.value:
                self.W.append(j)
                if self.enable_log:
                    if self.zero_base:
                        self.log.append([j])
                    else:
                        self.log.append([j + 1])
                self.set_S(j, S_state.Occupied)
                break
    
    def remove_car(self, car_id):
        self.set_S(self.W[car_id], S_state.Unavailable)
        self.W[car_id] = -1

    def fit_iter(self, car_id):
        if self.W[car_id] == -1:
            return -1
        end_flag = True
        cur_S = self.W[car_id]
        self.W[car_id] = -1
        for i in range(self.min_margin[car_id], np.max([cur_S, len(self.S_cfg) - cur_S])):
            right = cur_S + i
            if right < len(self.S_cfg):
                if self.O[right].value == S_state.Available.value:
                    self.set_S(right, S_state.Occupied)
                    self.W[car_id] = right
                    if self.enable_log:
                        if self.zero_base:
                            self.log[car_id].append(right)
                        else:
                            self.log[car_id].append(right + 1)
                    end_flag = False
                    break
                elif self.O[right].value == S_state.Occupied.value:
                    end_flag = False
            left = cur_S - i
            if left >= 0:
                if self.O[left].value == S_state.Available.value:
                    self.set_S(left, S_state.Occupied)
                    self.W[car_id] = left
                    if self.enable_log:
                        if self.zero_base:
                            self.log[car_id].append(left)
                        else:
                            self.log[car_id].append(left + 1)
                    end_flag = False
                    break
                elif self.O[left].value == S_state.Occupied.value:
                    end_flag = False
            

        if self.W[car_id] == -1:
            for i in np.arange(self.min_margin[car_id] - 1, 0, -1):
                right = cur_S + i
                if right < len(self.S_cfg):
                    if self.O[right].value == S_state.Available.value:
                        self.set_S(right, S_state.Occupied)
                        self.W[car_id] = right
                        if self.enable_log:
                            if self.zero_base:
                                self.log[car_id].append(right)
                            else:
                                self.log[car_id].append(right + 1)
                        end_flag = False
                        break
                    elif self.O[right].value == S_state.Occupied.value:
                        end_flag = False
                left = cur_S - i
                if left >= 0:
                    if self.O[left].value == S_state.Available.value:
                        self.set_S(left, S_state.Occupied)
                        self.W[car_id] = left
                        if self.enable_log:
                            if self.zero_base:
                                self.log[car_id].append(left)
                            else:
                                self.log[car_id].append(left + 1)
                        end_flag = False
                        break
                    elif self.O[left].value == S_state.Occupied.value:
                        end_flag = False
                
        if end_flag: 
            print("------complete------")
            self.complete = True
            if self.enable_log:
                print(self.log)
        
        return self.W[car_id]

if __name__ == "__main__":
    n = 24
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            D[i, j] = 7.5 * (j - i)
    D += D.T
    ori = D[:3, :]
    car_cfg = [
        {'v': 6, 'tt': 2, 'cw': 4, 'cv': 3, 'w': 6, 'min_R': 7.15},
        {'v': 4, 'tt': 2, 'cw': 4, 'cv': 3, 'w': 4, 'min_R': 7.15},
        {'v': 2, 'tt': 1, 'cw': 4, 'cv': 3, 'w': 2, 'min_R': 7.15},
    ]
    lenth = 300
    S = lenth * np.ones(n)
    des = np.zeros(ori.shape)
    k = 0
    IIG = IIG(D, ori, car_cfg, S, 7.5, des)
    dis = np.zeros(len(car_cfg)).tolist()
    step = 1
    w = [cfg['w'] for cfg in car_cfg]
    order = np.argsort(w)[::-1]
    while not IIG.is_complete():
        k += step
        if k == 250:
            # IIG.remove_car(2)
            # w[2] = 0
            # IIG.add_car({'v': 2, 'tt': 1, 'cw': 4, 'cv': 3, 'w': 6, 'min_R': 7.15}, D[10, :])
            # dis.append(0)
            # w.append(6)
            w[2] = 8
            order = np.argsort(w)[::-1]
            # IIG.set_S(4, S_state.Unavailable)
            # IIG.set_S(5, S_state.Unavailable)
            # IIG.set_S(6, S_state.Unavailable)
            # IIG.set_S(7, S_state.Unavailable)
        # if k == 300:
        #     IIG.set_S(4, S_state.Available)
        #     IIG.set_S(5, S_state.Available)
        #     IIG.set_S(6, S_state.Available)
        #     IIG.set_S(7, S_state.Available)
        for i in order:
            dis[i] += w[i] * step
            if dis[i] >= lenth:
                IIG.set_car_S(i, S_state.Unavailable)
                dis[i] = 0
                IIG.fit_iter(i)



