import numpy as np
from env.sac.core import SquashedGaussianMLPActor
import torch
import torch.nn as nn
import os

class trail_ctrl:
    def __init__(self,
                P_ct = 0.,
                D_ct = 0.,
                P_at = 0.,
                D_at = 0.,
                # P_D = 0.,
                # D_D = 0.,
                # P_X = 0.,
                # D_X = 0.,
                # P_t = 0.,
                # D_t = 0.
                ) -> None:
        self.P_ct = P_ct
        self.D_ct = D_ct
        self.P_at = P_at
        self.D_at = D_at
        # self.P_D = P_D
        # self.D_D = D_D
        # self.P_X = P_X
        # self.D_X = D_X
        # self.P_t = P_t
        # self.D_t = D_t
        # self.u_v = 0
        # self.u_p = 0

    def trail_follow(self, t, x_d, v_d, psi, x, v, direct = True, debug = False):
        e_ct = np.dot((x - x_d), t) * t - (x - x_d)
        v_t = v * np.array([np.cos(psi), np.sin(psi)])
        de_ct = np.dot(v_t, t) * t - v_t
        if direct:
            u_at = v_d * t
        else:
            if v_d - np.dot(v_t, t) > 0:
                e_at = (v_d - np.dot(v_t, t)) * t
            else:
                e_at = [0, 0]
            
            u_at = self.P_at * e_at + v_d * t
        u_ct = self.P_ct * e_ct + self.D_ct * de_ct

        # if v_d * np.dot(v_t, t) > 0:
        u = u_at + u_ct
        # else:
        #     u = u_at - u_ct
        
        u_v = np.linalg.norm(u)
        # print(u[0]+u[1]*1j)
        # u_p = np.angle(u[0]+u[1]*1j)
        u_p = np.arctan2(u[1], u[0])
        
        if debug:
            return u_v, u_p, np.linalg.norm(e_ct), abs(v - v_d)
        return u_v,u_p

class car_follow:
    def __init__(
            self, 
            P_vdes = 1, 
            D_vdes=0.5, 
            I_vdes = 0, 
            kp = 0.5, 
            ki = 1, 
            epsilon = 0.5, 
            k = 1, 
            model_path = "pi.pth", 
            V_max = 10, 
            algo = "slide"
        ):
        self.P_vdes=P_vdes
        self.D_vdes=D_vdes
        self.I_vdes=I_vdes
        self.diff=0
        self.itg = 0
        self.kp = kp
        self.ki = ki
        self.epsilon = epsilon
        self.k = k
        self.V_max = V_max
        self.algo = algo
        if model_path:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.pi = SquashedGaussianMLPActor(2, 1, hidden_sizes=(128,128,128), activation=nn.ReLU, act_limit = 1.0)
            path = os.path.abspath(__file__).split(os.sep)[:-1]
            model_path = os.path.join(os.sep, *path,'sac', model_path)
            self.pi.load_state_dict(torch.load(model_path, map_location = self.device))
            self.pi.eval()
        
        
    #     return t_d,x_start,v_d
    def update_state(self, t, leader_pos, follower_pos, m_a, T, leader_v, follower_v):
        # P, I, D
        t_diff = np.dot(t, leader_pos - follower_pos) + m_a
        self.itg += t_diff * T
        es_dot = np.dot(leader_v, t) - np.dot(follower_v, t)
        return t_diff, self.itg, es_dot

    def follow(self,t,leader_ori,m_c,leader_pos,leader_v,leader_psi,leader_v_set, follower_pos, follower_v,follower_psi, T, m_a=0):
        # direction
        t_d = t

        # follower reference point
        x_start = leader_ori + np.dot(m_c, [t[1], -t[0]])
        
        # velosity vectors
        follower_v_t = follower_v * np.array([np.cos(follower_psi), np.sin(follower_psi)])
        leader_v_t = leader_v * np.array([np.cos(leader_psi), np.sin(leader_psi)])

        # get PID state
        t_diff, itg, es_dot = self.update_state(t, leader_pos, follower_pos, m_a, T, leader_v_t, follower_v_t)
        
        # calculate the desired project velosity using the selected algorithm
        
        # PID controller
        if self.algo == "PID" or self.algo == "pid":
            v_d = leader_v_set + self.P_vdes * t_diff + self.D_vdes * es_dot + self.I_vdes * itg
        # sliding mode controller
        elif self.algo == "slide":
            tmp = es_dot + self.kp * t_diff + self.ki * itg
            v_d = leader_v_set + (self.kp * es_dot + self.ki * t_diff + self.epsilon * np.sign(tmp) + self.k * tmp) * T
        # RL agent
        elif self.algo == "sac" or self.algo == 'SAC':
            state = np.array([t_diff, es_dot]) / self.V_max
            with torch.no_grad():
                v_d, _ = self.pi(torch.as_tensor(state, dtype=torch.float32), True, False)
            v_d = v_d.cpu().numpy()
            v_d *= self.V_max
            v_d += np.dot(leader_v_t, t) #leader_v_set

        # output clip
        if v_d < 0:
            v_d = 0
        elif v_d > self.V_max:
            v_d = self.V_max
        
        # return the desired velosity vector
        return t_d, x_start, v_d
