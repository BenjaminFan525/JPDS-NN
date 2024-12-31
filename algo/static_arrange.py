import numpy as np
import random
import copy
import matplotlib.pyplot as plt

def task_manage(points, width):
    for idx, point in enumerate(points):
        pass

def fit(D, ori, des, car_cfg, tS, T, type='s'):
    '''
    D: distance matrix (N x N) or (N x N x 2 x 2)
    ori: original distances (m x N) or (m x N x 2 x 2)
    des: destination distances (m x N) or (m x N x 2 x 2)
    car_cfg: car config (m)
    tS: operation time for each car for each line (m x N)
    T: an arrangement [[],...,[]]
    type: 's' for distance, 't' for time, 'c' for cost
    '''
    assert D.shape[0] == D.shape[1]
    entry = False
    if len(D.shape) == 4:
        assert D.shape[2] == D.shape[3] == 2
        entry = True

    num_car = ori.shape[0]
    s = []
    tv = []
    for a in range(num_car):
        if len(T[a]) > 0:
            if entry:
                s_tmp = ori[a, T[a][0][0], T[a][0][1], 0]
            else:
                s_tmp = ori[a, T[a][0]]
            for i in range(len(T[a])-1):
                if entry:
                    s_tmp += D[T[a][i][0], T[a][i + 1][0], 1 - T[a][i][1], T[a][i + 1][1]]
                else:
                    s_tmp += D[T[a][i], T[a][i + 1]]
            if entry:
                s_tmp += des[a, T[a][-1][0], 1 - T[a][-1][1], 0]
            else:
                s_tmp += des[a, T[a][-1]]
            s.append(s_tmp)
            tv.append(s_tmp / car_cfg[a]['vv'])
        else:
            s.append(0)
            tv.append(0)
    s = np.array(s)
    tv = np.array(tv)
    output = [np.sum(s)]
    # if type == 's':
    #     return 1 / np.sum(s)
    
    select = np.zeros(ori.shape[:2])
    for a in range(num_car):
        if len(T[a]) > 0:
            if entry:
                select[a][np.array(T[a])[:, 0]] = 1
            else:
                select[a][T[a]] = 1
        tv[a] += (np.max([(len(T[a]) - 1), 0]) * car_cfg[a]['tt'])
    
    tw = np.sum(tS * select, axis=1)
    output.append(np.max(tv + tw))
    # if type == 't':
    #     return 1 / np.max(tv + tw)
    
    cv = [cfg['cv'] for cfg in car_cfg]
    cw = [cfg['cw'] for cfg in car_cfg]

    output.append((np.array(cv) @ tv + np.array(cw) @ tw).item())
    if type == 's':
        return 1 / output[0]
    elif type == 't':
        return 1 / output[1]
    else:
        return 1 / output[2]
    # return 1 / (np.array(cv) @ tv + np.array(cw) @ tw).item()

class MGGA:
    _defaults = {
        "gen_size"    : 200,

        "max_iter"    : 200,

        "pc"          : 0.6,

        "p_op"        : 0.5,

        "p_ex"        : 0.5,

        "p_mv"        : 0.6,

        "p_inv"       : 0.5,

        "p_order"     : 0.8,

        "p_e_order"   : 0.8,

        "f"           : 't',

        "order"       : False,

        "render"      : False,

        "entry"       : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def optimize(self, D, ori, car_cfg, S_cfg, des = None, line_field_idx = None):
        if len(D.shape) == 4:
            assert D.shape[2] == D.shape[3] == 2
            self.entry = True
        num_car, num_target = ori.shape[:2]
        if des is None:
            des = np.zeros(ori.shape)
        tS = np.zeros((num_car, len(S_cfg)))
        for i in range(num_car):
            tS[i] = S_cfg / car_cfg[i]['vw']
        gen = []
        f_parents = []
        best_f = 0
        best_id = 0
        for i in range(self.gen_size):
            if self.entry:
                gen.append({'tar': [[x, int(np.round(random.random()))] for x in range(num_target)], 'split': np.random.choice(range(num_target + 1), num_car - 1, replace = True).tolist()})
            else:
                gen.append({'tar': list(range(num_target)), 'split': np.random.choice(range(num_target + 1), num_car - 1, replace = True).tolist()})
            random.shuffle(gen[i]['tar'])
            gen[i]['split'].sort()
            # if self.order:
            #     gen[i] = self.gen_sort(gen[i], line_field_idx)
            f_parents.append(fit(D, ori, des, car_cfg, tS, self.decode(gen[i]), self.f))
            if f_parents[i] > best_f:
                best_f = f_parents[i]
                best_id = i

        best_fs = []        
        for iter in range(self.max_iter):
            children = []
            f_children = []
            worst_f = np.inf
            worst_id = 0
            c_best_f = 0
            c_best_id = 0
            
            for i in range(self.gen_size):
                ###### choose & cross #######
                if random.random() < self.pc:
                    child = self.cross(random.choices(gen, weights=f_parents, k=2))
                else:
                    child = copy.deepcopy(random.choices(gen, weights=f_parents, k=1)[0])

                ###### mutate ##########
                if random.random() < self.p_mv:
                    child = self.m_mv(child)
                if random.random() < self.p_ex:
                    child = self.m_ex(child)
                if self.entry:
                    if random.random() < self.p_inv:
                        child = self.m_entry_inv(child)
                # if not self.order:
                if random.random() < self.p_op:
                    child = self.m_op(child)
                if random.random() < self.p_order:
                    child = self.gen_sort(child, line_field_idx)
                if self.entry:
                    if random.random() < self.p_e_order:
                        child = self.m_entry_sort(child, D, ori)

                children.append(child)
                f_children.append(fit(D, ori, des, car_cfg, tS, self.decode(child), self.f))
                if f_children[i] > c_best_f:
                    c_best_f = f_children[i]
                    c_best_id = i
                if f_children[i] < worst_f:
                    worst_f = f_children[i]
                    worst_id = i
            
            if c_best_f < best_f:
                children[worst_id] = gen[best_id]
                f_children[worst_id] = f_parents[best_id]
                best_id = worst_id
            else:
                best_id = c_best_id
                best_f = c_best_f
            best_fs.append(1 / best_f)
            gen = children
            f_parents = f_children

            if self.render and iter % 5 == 0:
                color = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
                plt.subplot(1,2,1)
                plt.cla()
                curr_T = self.decode(gen[best_id])
                for idx, tr in enumerate(curr_T):
                    if len(tr) == 0:
                        continue
                    x_ax = D[0, np.array(tr)[:, 0], 0, 0] if self.entry else D[0, tr]
                    x_ax = np.array(list(zip(x_ax, x_ax))).flatten()
                    y_ax_0 = np.zeros(len(tr))
                    y_ax_s = S_cfg[np.array(tr)[:, 0]] if self.entry else S_cfg[tr]
                    y_ax = np.array(list(zip(y_ax_0, y_ax_s)))
                    mask = [int(mask_i % 2) == 1 for mask_i in range(len(y_ax))]
                    y_ax[mask] = np.fliplr(y_ax[mask])
                    plt.plot(x_ax, y_ax.flatten(), color[idx])
                    plt.xlabel("x(m)")
                    plt.ylabel("y(m)")
                plt.subplot(1,2,2)
                plt.cla()
                plt.plot(range(len(best_fs)), best_fs)
                plt.xlabel("iter")
                plt.ylabel("best " + self.f)
                plt.pause(0.05)
        
        return self.decode(gen[best_id]), 1 / best_f, best_fs
    
    def gen_sort(self, gen, line_field_idx):
        def entry_sort(seq, entry: int):
            seq.sort(key = lambda x : x[0])
            for idx in range(len(seq)):
                if bool(idx % 2):
                    seq[idx][1] = 1 - entry
                else:
                    seq[idx][1] = entry
            return seq
        
        T = self.decode(gen)
        for j, arrange in enumerate(T):
            if len(arrange) == 0:
                continue
            
            if line_field_idx is None:
                if not self.entry: # 不考虑入口点
                    T[j].sort()
                    return
                T[j] = entry_sort(arrange, T[j][0][1])
            else:
                start = 0
                end = 1
                while end < len(arrange):
                    if line_field_idx[arrange[end][0]] != line_field_idx[arrange[start][0]]:                    
                        T[j][start: end] = entry_sort(T[j][start: end], T[j][start][1])
                        start = end
                    end += 1
                T[j][start:] = entry_sort(T[j][start: end], T[j][start][1])
        return self.encode(T)

    def decode(self, gen):
        T = []
        start = 0
        for end in gen['split']:
            T.append(gen['tar'][start:end])
            start = end
        T.append(gen['tar'][start:])
        return T

    def encode(self, T):
        child_ = {'tar':[], 'split': []}
        for idx, tar in enumerate(T):
            child_['tar'] += tar
            if idx < len(T) - 1:
                if idx == 0:
                    child_['split'].append(len(tar))
                else:
                    child_['split'].append(child_['split'][idx - 1] + len(tar))
        return child_
        
    def cross(self, parents):
        oder = list(range(len(parents[0]['split']) + 1))
        random.shuffle(oder)
        T0 = copy.deepcopy(self.decode(parents[0]))
        T1 = copy.deepcopy(self.decode(parents[1]))
        child = []
        for i in range(len(T0)):
            child.append([])

        for i in oder:
            if random.random() < 0.5:
                child[i] = T0[i]
                T0[i] = []
                for j in child[i]:
                    for k in T1:
                        if self.entry:
                            idx = 0
                            found = False
                            while idx < len(k):
                                if j[0] == k[idx][0]:
                                    k.pop(idx)
                                    found = True
                                    break
                                idx += 1
                            if found:
                                break
                        else:
                            if j in k:
                                k.remove(j)
                                break
            else:
                child[i] = T1[i]
                T1[i] = []
                for j in child[i]:
                    for k in T0:
                        if self.entry:
                            idx = 0
                            found = False
                            while idx < len(k):
                                if j[0] == k[idx][0]:
                                    k.pop(idx)
                                    found = True
                                    break
                                idx += 1
                            if found:
                                break
                        else:
                            if j in k:
                                k.remove(j)
                                break
        remain = []
        for i in range(len(T0)):
            remain += T0[i]
        
        for i in remain:
            idx = random.randint(0, len(oder) - 1)
            place = random.randint(0, len(child[idx]))
            child[idx].insert(place, i)
            
        return self.encode(child)
                
    def m_ex(self, gen):
        T = self.decode(gen)
        op = []
        sp = []
        for t in T:
            if len(t) == 0:
                op.append([])
                sp.append(-1)
            else:
                j = random.choice(range(len(t)))
                op.append([t[j]])
                sp.append(j)
        random.shuffle(op)
        for i, t in enumerate(T):
            T[i] = t[:sp[i]] + op[i] + t[(sp[i] + 1) :]
        
        return self.encode(T)

    def m_op(self, gen):
        split = [0] + gen['split'] + [len(gen['tar'])]
        op = [i for i in range(len(split) - 1) if split[i + 1] - split[i] > 1]
        i = random.choice(op)
        sp = random.sample(range(split[i], split[i + 1]), 2)
        sp.sort()
        tmp = gen['tar'][sp[0] : (sp[1] + 1)]
        tmp.reverse()
        gen['tar'][sp[0] : (sp[1] + 1)] = tmp
        return gen

    def m_mv(self, gen):
        T = self.decode(gen)
        op = random.sample(range(len(T)), 2)
        if len(T[op[0]]) > 0:
            i = random.choice(T[op[0]])
            T[op[1]].insert(random.choice(range(len(T[op[1]]) + 1)), i)
            T[op[0]].remove(i)
            return self.encode(T)
        else:
            return gen
    
    def m_entry_inv(self, gen):
        T = self.decode(gen)
        idx = random.choice(range(len(T)))
        if len(T[idx]) > 0:
            T[idx] = [[x[0], 1 - T[idx][0][1]] for x in T[idx]]
        return self.encode(T)

    def m_entry_sort(self, gen, D, ori):
        Ts = self.decode(gen)
        for idx, T in enumerate(Ts):
            if len(T) == 0:
                continue
            T[0][1] = np.argmin(ori[idx, T[0][0], :, 0])
            for idx_t in range(1, len(T)):
                T[idx_t][1] = np.argmin(D[T[idx_t - 1][0], 
                                          T[idx_t][0], 
                                          1 - T[idx_t - 1][1]])
        return self.encode(Ts)
        
if __name__ == "__main__":
    from ia.utils import load_dict
    GA = MGGA(f = 't', order = True, render = False)
    # gen0 = {'tar': [0,1,2,3,4,5], 'split': [0,3]}
    # gen1 = {'tar': [5,2,4,0,3,1], 'split': [2,6]}
    # T = GA.decode(gen0)
    # gen = GA.encode(T)
    # gen = GA.m_mv(gen0)
    # D = np.array([
    #     [0, 1, 2, 3, 4, 5],
    #     [1, 0, 1, 2, 3, 4],
    #     [2, 1, 0, 1, 2, 3],
    #     [3, 2, 1, 0, 1, 2],
    #     [4, 3, 2, 1, 0, 1],
    #     [5, 4, 3, 2, 1, 0]
    # ])
    n = 24
    # D = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         D[i, j] = 7.5 * (j - i)
    # ori = D[:3]

    # D += D.T
    D = np.zeros((n, n, 2, 2))
    for i in range(n):
        for j in range(n):
            D[i, j, 0, 0] = D[i, j, 1, 1] = 7.5 * (j - i)
            D[i, j, 1, 0] = D[i, j, 0, 1] = 7.5 * (j - i) + 50
    ori = D[:3, :, 0]
    car_cfg = [
        {'vv': 6, 'tt': 2, 'cw': 8, 'cv': 3, 'vw': 6, 'min_R': 7.15},
        {'vv': 4, 'tt': 2, 'cw': 4, 'cv': 3, 'vw': 4, 'min_R': 7.15},
        {'vv': 4, 'tt': 2, 'cw': 3, 'cv': 3, 'vw': 2, 'min_R': 7.15},
    ]
    S = 300 * np.ones(n)
    des = np.zeros(ori.shape)
    import time
    tic = time.time()
    T, best, log = GA.optimize(D, ori, car_cfg, S)
    print(time.time() - tic)
    print(T)
    print(best)
    # gen = GA.encode(T)
    num_car = len(car_cfg)
    tS = np.zeros((num_car, len(S)))
    for i in range(num_car):
        tS[i] = S / car_cfg[i]['vw']
    # print(1/GA.fit(D, ori, des, car_cfg, tS, gen))
    T = [[0,3,5,7,9,11,13,15,17,19,21,23],
         [1,4,6,10,12,16,18,22],
         [2,8,14,20]]
    gen = GA.encode(T)
    print(GA.decode(gen))
    print(1/fit(D, ori, des, car_cfg, tS, T, GA.f))
    T = [[12,13,14,15,16,17,18,19,20,21,22,23],
         [4,5,6,7,8,9,10,11],
         [0,1,2,3]]
    gen = GA.encode(T)
    print(GA.decode(gen))
    print(1/fit(D, ori, des, car_cfg, tS, T, GA.f))
    # plt.plot(range(len(log)), log)
    # plt.xlabel("iter")
    # plt.ylabel("best f")
    plt.show()