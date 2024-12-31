import numpy as np
import torch

def fit(D, ori, des, car_cfg, tS, T, type='s', tS_t=True, dense=False):
    '''
    D: distance matrix (N x N) or (N x N x 2 x 2)
    ori: original distances (m x N) or (m x N x 2 x 2)
    des: destination distances (m x N) or (m x N x 2 x 2)
    car_cfg: car config (m)
    tS: operation time for each car for each line or length of each line (m x N)
    T: an arrangement [[],...,[]]
    tS_t: if True, tS represents the operation time, otherwise the length of each line
    type: 's' for distance, 't' for time, 'c' for oil cost, 'all' for all
    dense: return every step's cost
    '''
    assert D.shape[0] == D.shape[1]
    entry = False
    if len(D.shape) == 4:
        assert D.shape[2] == D.shape[3] == 2
        entry = True
    
    num_car = ori.shape[0]
    if not tS_t:
        lS = np.zeros((num_car, len(tS)))
        for i in range(num_car):
            lS[i] = tS / car_cfg[i]['vw']
        tS = lS

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
    output = [np.sum(s).item()]
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
    output.append(np.max(tv + tw).item())
    # if type == 't':
    #     return 1 / np.max(tv + tw)
    
    cv = [cfg['cv'] for cfg in car_cfg]
    cw = [cfg['cw'] for cfg in car_cfg]

    output.append((np.array(cv) @ tv + np.array(cw) @ tw).item())
    if type == 's':
        return 1 / output[0]
    elif type == 't':
        return 1 / output[1]
    elif type == 'c':
        return 1 / output[2]
    else:
        return output

def dense_fit(D, ori, des, car_cfg, lS, T, total_time = False):
    '''
    D: distance matrix (N x N) or (N x N x 2 x 2)
    ori: original distances (m x N) or (m x N x 2 x 2)
    des: destination distances (m x N) or (m x N x 2 x 2)
    car_cfg: car config (m)
    lS: length of each line (m)
    T: an arrangement [[],...,[]]
    total_time: use total t as t if True
    '''
    assert D.shape[0] == D.shape[1]
    entry = False
    if len(D.shape) == 4:
        assert D.shape[2] == D.shape[3] == 2
        entry = True

    num_car = ori.shape[0]
    # tS = np.zeros((num_car, len(lS)))
    # for i in range(num_car):
    #     tS[i] = lS / car_cfg[i]['vw']

    s = []
    t = []
    c = []
    total_t = []
    t_max = 0
    for a in range(num_car):
        cur_t = 0
        if len(T[a]) > 0:
            if entry:
                s += [ori[a, T[a][0][0], T[a][0][1], 0]]
                cur_t += ori[a, T[a][0][0], T[a][0][1], 0] / car_cfg[a]['vv'] + \
                    lS[T[a][0][0]] / car_cfg[a]['vw'] + car_cfg[a]['tt']
                total_t.append(ori[a, T[a][0][0], T[a][0][1], 0] / car_cfg[a]['vv'] + \
                    lS[T[a][0][0]] / car_cfg[a]['vw'] + car_cfg[a]['tt'])
                if cur_t > t_max:
                    t.append(cur_t - t_max)
                    t_max = cur_t
                else:
                    t.append(0)
                c.append(ori[a, T[a][0][0], T[a][0][1], 0] / car_cfg[a]['vv'] * car_cfg[a]['cv'] + \
                          lS[T[a][0][0]] / car_cfg[a]['vw'] * car_cfg[a]['cw'])
            else:
                s += [ori[a, T[a][0]]]
            for i in range(len(T[a])-1):
                if entry:
                    s += [D[T[a][i][0], T[a][i + 1][0], 1 - T[a][i][1], T[a][i + 1][1]]]
                    cur_t += D[T[a][i][0], T[a][i + 1][0], 1 - T[a][i][1], T[a][i + 1][1]] / car_cfg[a]['vv'] + \
                        lS[T[a][i + 1][0]] / car_cfg[a]['vw'] + car_cfg[a]['tt']
                    total_t.append(D[T[a][i][0], T[a][i + 1][0], 1 - T[a][i][1], T[a][i + 1][1]] / car_cfg[a]['vv'] + \
                        lS[T[a][i + 1][0]] / car_cfg[a]['vw'] + car_cfg[a]['tt'])
                    if cur_t > t_max:
                        t.append(cur_t - t_max)
                        t_max = cur_t
                    else:
                        t.append(0)
                    c.append(D[T[a][i][0], T[a][i + 1][0], 1 - T[a][i][1], T[a][i + 1][1]] / car_cfg[a]['vv']  * car_cfg[a]['cv']+ \
                        lS[T[a][i + 1][0]] / car_cfg[a]['vw'] * car_cfg[a]['cw'])
                else:
                    s += [D[T[a][i], T[a][i + 1]]]
            if entry:
                s += [des[a, T[a][-1][0], 1 - T[a][-1][1], 0]]
                cur_t += des[a, T[a][-1][0], 1 - T[a][-1][1], 0] / car_cfg[a]['vv']
                total_t.append(des[a, T[a][-1][0], 1 - T[a][-1][1], 0] / car_cfg[a]['vv'])
                if cur_t > t_max:
                    t.append(cur_t - t_max)
                    t_max = cur_t
                else:
                    t.append(0)
                c.append(des[a, T[a][-1][0], 1 - T[a][-1][1], 0] / car_cfg[a]['vv'] * car_cfg[a]['cv'])
            else:
                s += [des[a, T[a][-1]]]
        else:
            s.append(0)
            t.append(0)
            c.append(0)
            total_t.append(0)
    s[-2] += s[-1]
    t[-2] += t[-1]
    c[-2] += c[-1]
    total_t[-2] += total_t[-1]
    s = torch.tensor(np.array(s[:-1]), dtype=torch.float32)
    t = torch.tensor(np.array(t[:-1]), dtype=torch.float32)
    c = torch.tensor(np.array(c[:-1]), dtype=torch.float32)
    total_t = torch.tensor(np.array(total_t[:-1]), dtype=torch.float32)
    return {'s': s, 't': total_t if total_time else t, 'c': c}


def decode(gen):
    T = [[] for _ in range(len(gen['split']))]
    start = 0
    for end in gen['split']:
        T[end[1]] = gen['tar'][start:end[0]]
        start = end[0]
    return T

def encode(T):
    child_ = {'tar':[], 'split': []}
    for idx, tar in enumerate(T):
        child_['tar'] += tar
        if idx < len(T) - 1:
            if idx == 0:
                child_['split'].append(len(tar))
            else:
                child_['split'].append(child_['split'][idx - 1] + len(tar))
    return child_

def choice2T(choice, target_num):
    T = [[] for _ in range(target_num + 1)]
    for idx, tgt in enumerate(choice):
        T[tgt - 1].append(idx)
    return T

