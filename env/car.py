from matplotlib import markers
import numpy as np
import threading
import yaml
from enum import Enum
import matplotlib.pyplot as plt

import os
# import sys
# sys.path.append(__file__.replace('env/car.py',''))
from env.trail_v_2 import trail_ctrl

# drive model, use your ideal mode for control
DRIVE_MODE=Enum("DRIVE_MODE",('MANUAL','AUTO_ACC_OMEGA','AUTO_VEL_OMEGA','AUTO_VEL_PSI','DIRECT', 'DIRECT_PSI'))

# class pose
class state_:
    def __init__(self,state_ini=np.zeros(6)):
        self.x=state_ini[0]
        self.y=state_ini[1]
        self.v=state_ini[2]
        self.a=state_ini[3]
        self.psi=state_ini[4]
        self.w=state_ini[5]
    
    def setposition(self,p_input):
        self.x=p_input[0]
        self.y=p_input[1]
    
    def setpose(self,p_input):
        self.x=p_input[0]
        self.y=p_input[1]
        self.psi=p_input[2]
    
    def position(self):
        return [self.x,self.y]

class car:
    def __init__(self,car_model,T=0.01,state_ini=np.zeros(6),log=False, 
                 noise = None, ctrl_period = 0.02, debug = False):
        self.T=T
        self.D=0.
        self.D_=0.
        self.X=0.
        self.X_=0.
        self.theta=0.
        self.theta_=0.
        self.a_des=0.
        self.v_des=0.
        self.w_des=0.
        self.psi_des=0.
        self.state=state_(state_ini)
        self.ea=0
        self.ew=0
        self.ev=0
        self.ep=0

        path = os.path.abspath(__file__).split(os.sep)[:-2]
        cfg_path = os.path.join(os.sep, *path,'config',"car.yaml")

        file = open(cfg_path, 'r', encoding="utf-8")
        file_data = file.read()
        file.close()
        data = yaml.safe_load(file_data)[car_model]

        for key, value in data.items():
            setattr(self, key, value)
        # self.f=data['f']
        # self.t_D=data['t_D']
        # self.t_X=data['t_X']
        # self.t_theta=data['t_theta']
        # self.k_D=data['k_D']
        # self.M=data['M']
        # self.L=data['L']
        # self.K=data['K']
        # self.L_f=data['L_f']
        # self.L_b=data['L_b']
        # self.D_max=data['D_max']
        # self.X_max=data['X_max']
        # self.theta_max=data['theta_max']
        # self.inner_PID=data['inner_PID']
        self.R_min = self.L / np.tan(self.theta_max)
        self.D_min = np.sqrt(self.f / self.k_D)
        self.V_max = self.D_max / self.f

        self.trail_ctrler = trail_ctrl()
        for key, value in self.trail_ctrl_cfg.items():
            setattr(self.trail_ctrler, key, value)

        self.realtime_enable=False

        self.noise = noise

        self.ctrl_period = ctrl_period

        self.physical_step = int(ctrl_period / T)

        self.ctrl_period = self.physical_step * T 

        self.debug = debug
        if debug:
            self.total_ect = 0
            self.total_ev = 0

    def reset(self, state_ini=np.zeros(6)):
        self.D=0.
        self.D_=0.
        self.X=0.
        self.X_=0.
        self.theta=0.
        self.theta_=0.
        self.a_des=0.
        self.v_des=0.
        self.w_des=0.
        self.psi_des=0.
        self.state=state_(state_ini)
        self.ea=0
        self.ew=0
        self.ev=0
        self.ep=0

    def position(self):
        return np.array([self.state.x,self.state.y])
    
    def v(self):
        return self.state.v
    
    def a(self):
        return self.state.a
    
    def w(self):
        return self.state.w
    
    def psi(self):
        return self.state.psi

    def realtimeEnable(self):
        self.realtime_enable=True
        self.realtime()

    def realtimeDisable(self):
        self.realtime_enable=False

    def realtime(self):
        if self.realtime_enable:
            self.t = threading.Timer(self.T, self.update_state(DRIVE_MODE.BASIC))
            self.t.start()

    def update_state(self, mode):
        if mode.value != DRIVE_MODE.DIRECT.value and mode.value != DRIVE_MODE.DIRECT_PSI.value:
            if mode.value != DRIVE_MODE.MANUAL.value:
                if mode.value == DRIVE_MODE.AUTO_ACC_OMEGA.value or mode.value == DRIVE_MODE.AUTO_VEL_OMEGA.value:
                    if self.state.v != 0:
                        theta_des=np.arctan(self.w_des*self.L/self.state.v)
                        self.theta = theta_des+self.inner_PID['P_w']*(theta_des-self.theta_)#+self.inner_PID['D_w']*(theta_des-self.theta_-self.ew)/self.ctrl_period
                        self.ew = theta_des-self.theta_
                else:
                    psi_diff = self.psi_des-self.state.psi
                    if abs(psi_diff)>np.pi:
                        psi_diff -= np.sign(psi_diff)*2*np.pi
                    self.theta = self.inner_PID['P_p']*(psi_diff)+self.inner_PID['D_p']*(psi_diff-self.ep)/self.ctrl_period
                    self.ep = psi_diff
                
                # if abs(self.theta) > self.theta_max:
                #     self.theta = np.sign(self.theta)*self.theta_max

                if mode.value == DRIVE_MODE.AUTO_VEL_PSI.value or mode.value == DRIVE_MODE.AUTO_VEL_OMEGA.value:
                    vctrl = self.inner_PID['P_v']*(self.v_des-self.state.v)+self.inner_PID['D_v']*(self.v_des-self.state.v-self.ev)/self.ctrl_period+self.f*self.v_des
                    if vctrl>0:
                        self.D=vctrl * self.f
                        self.X=0.
                    else:
                        self.D=0.
                        self.X=-vctrl * self.M
                    self.ev = self.v_des-self.state.v
                else:
                    vctrl = self.inner_PID['P_a']*(self.a_des-self.state.a)+self.inner_PID['D_a']*(self.a_des-self.state.a-self.ea)/self.ctrl_period+self.f*self.v_des
                    if vctrl>0:
                        self.D=vctrl * self.f
                        self.X=0.
                    else:
                        self.D=0.
                        self.X=-vctrl * self.M
                    self.ea = self.a_des-self.state.a
                
                # if self.D > self.D_max:
                #     self.D = self.D_max
                # if self.X > self.X_max:
                #     self.X = self.X_max
            
        for _ in range(self.physical_step):
            self.phisic_step(mode)
            
        self.realtime()
    
    # Dynamic and kinetic models 
    # 动力学与运动学模型
    def phisic_step(self, mode):
        # 噪声
        if self.noise is not None:
            noise = np.random.multivariate_normal(np.zeros(6), self.noise)
        else:
            noise = np.zeros(6)
        noise *= (self.state.v if self.state.v > 0.1 else 0)
        
        if mode.value != DRIVE_MODE.DIRECT.value and mode.value != DRIVE_MODE.DIRECT_PSI.value:
            # 油门一阶模型
            self.D_ += 1/self.t_D*(self.D-self.D_)*self.T
            if self.D_ > self.D_max:
                self.D_ = self.D_max
            
            # 刹车一阶模型
            self.X_ += 1/self.t_X*(self.X-self.X_)*self.T
            if self.X_ > self.X_max:
                self.X_ = self.X_max
            
            # 方向盘一阶模型
            self.theta_ += 1/self.t_theta*(self.theta-self.theta_)*self.T
            if abs(self.theta_) > self.theta_max:
                self.theta_ = np.sign(self.theta_)*self.theta_max
            
            # 驱动力
            if self.state.v > 0.01:
                self.F_T = self.D_ / self.state.v
            else:
                self.F_T = self.k_D * self.D_**2

            # 阻力
            if self.state.v > 0 or self.F_T >= (self.X_ + self.f):
                self.F_f =  self.X_ + self.f  # 动摩擦
            else:
                self.F_f = self.F_T           # 静摩擦

            self.state.a = 1/self.M * (self.F_T-self.F_f) + noise[3]
            self.state.w = self.state.v / self.L * np.tan(self.theta_) + noise[5]    
        else:
            self.state.v = min(self.v_des, self.V_max)
            if mode.value == DRIVE_MODE.DIRECT_PSI.value:
                psi_diff = self.psi_des-self.state.psi
                if psi_diff > np.pi:
                    psi_diff -= 2*np.pi
                elif psi_diff <= -np.pi:
                    psi_diff += 2 * np.pi
                
                # 以最大角速度跟踪期望方向
                self.state.w = np.sign(psi_diff) * self.state.v / self.R_min
            else:
                self.state.w = self.w_des

        v0 = self.state.v
        if mode.value != DRIVE_MODE.DIRECT.value:
            self.state.v += self.state.a*self.T + noise[2]
            if self.state.v < 0:
                self.state.v = 0
        
        self.ave_v = (v0 + self.state.v) / 2
        self.state.x += self.ave_v * np.cos(self.state.psi) * self.T + noise[0]
        self.state.y += self.ave_v * np.sin(self.state.psi) * self.T + noise[1]
        self.state.psi += self.state.w * self.T + noise[4]
        while self.state.psi > np.pi:
            self.state.psi -= 2 * np.pi
        while self.state.psi <= -np.pi:
            self.state.psi += 2 * np.pi

    def follow_trail(self, t, x_d, v_d, mode = "physical", direct = True, follow = True, inner_loop_times = 1):
        if self.debug:
            self.v_des, self.psi_des, ect, ev = self.trail_ctrler.trail_follow(
                                                                    t,
                                                                    x_d,
                                                                    v_d,
                                                                    self.psi(),
                                                                    self.position(),
                                                                    self.v(),
                                                                    direct,
                                                                    True
                                                                )
            
            self.total_ect += ect
            self.total_ev += ev
        else:
            self.v_des, self.psi_des = self.trail_ctrler.trail_follow(
                                                                    t,
                                                                    x_d,
                                                                    v_d,
                                                                    self.psi(),
                                                                    self.position(),
                                                                    self.v(),
                                                                    direct
                                                                )

        if follow:
            if mode == "physical":
                drive_mode = DRIVE_MODE.AUTO_VEL_PSI
            else:
                drive_mode = DRIVE_MODE.DIRECT_PSI
            
            for _ in range(inner_loop_times):
                self.update_state(drive_mode)

    def stop(self):
        self.v_des = 0

    #mode = 0, 1, 2, 0 for single dot, 1 for simple rectangle, 2 for whole car
    def plot(self, color = 'b', mode = 0, ax = None, return_bound = False, show = True):
        if show:
            if ax is None:
                _, ax = plt.subplots()
                ax.axis('equal')

        f1, f2 = None, None
        if mode == 1:
            car_shape = np.array([
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
                [self.state.x+np.sin(self.state.psi)*self.K / 2 , self.state.y-np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L + np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L - np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L - np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L + np.cos(self.state.psi)*self.K / 2],
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
            ])
            if show:
                f1 = ax.plot(car_shape[:,0], car_shape[:,1], color = color)
                f2 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
            if return_bound:
                return (f1, f2), car_shape[:4]
        elif mode == 2:
            car_shape = np.array([
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
                [self.state.x+np.sin(self.state.psi)*self.K / 2 , self.state.y-np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L - np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L + np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L + np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L - np.cos(self.state.psi)*self.K / 2],
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
            ])
            if show:
                f1 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
                f2 = ax.plot(car_shape[:,0],car_shape[:,1], color = color)
            if return_bound:
                return (f1, f2), car_shape[:4]
        else:
            if show:
                f1 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
            return (f1, f2)

class Robot(car):
    def __init__(self, robo_model: dict, T=0.01, state_ini=np.zeros(6), log=False, noise=None, ctrl_period=0.02, debug = False):
        '''
        A config should contain
            -- car_model
            -- working_width (m)
            -- cw: cost working (L/s)
            -- cv: cost driving (L/s)
            -- tt: average turning time (s)
            -- vw: working speed (m/s)
            -- vv: driving speed (m/s)
        '''     
        for key, value in robo_model.items():
            setattr(self, key, value)

        super().__init__(self.car_model, T, state_ini, log, noise, ctrl_period, debug)
        
        self.working = False
        self.total_cost = 0.
        self.total_distance = 0.
        if self.debug:
            self.tv = 0.
            self.tw = 0.
            self.working_distance = 0.
            self.driving_distance = 0.

    def phisic_step(self, mode):
        super().phisic_step(mode)
        if self.working:
            self.total_cost += self.cw * self.T * int(self.ave_v > 0)
            if self.debug:
                self.tw += self.T * int(self.ave_v > 0)
                self.working_distance += self.ave_v * self.T
        else:
            self.total_cost += self.cv * self.T * int(self.ave_v > 0)
            if self.debug:
                self.tv += self.T * int(self.ave_v > 0)
                self.driving_distance += self.ave_v * self.T
        
        self.total_distance += self.ave_v * self.T
    
    def plot(self, color = 'b', mode = 0, ax = None, return_bound = False, show = True, label=None):
        if show:
            if ax is None:
                _, ax = plt.subplots()
                ax.axis('equal')

        f1, f2 = None, None
        if mode == 1:
            car_shape = np.array([
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
                [self.state.x+np.sin(self.state.psi)*self.K / 2 , self.state.y-np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L + np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L - np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L - np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L + np.cos(self.state.psi)*self.K / 2],
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
            ])
            if show:
                f1 = ax.plot(car_shape[:,0], car_shape[:,1], color = color, label=label)
                # f2 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
                ax.plot([self.state.x-np.sin(self.state.psi)*self.working_width / 2, self.state.x+np.sin(self.state.psi)*self.working_width / 2],
                        [self.state.y+np.cos(self.state.psi)*self.working_width / 2, self.state.y-np.cos(self.state.psi)*self.working_width / 2],
                        color = color, 
                        linewidth = 1)
                # if self.working:
                #     # ax.add_patch(
                #     #     plt.Rectangle(
                #     #         (car_shape[0, 0], car_shape[0, 1]), 
                #     #         self.K, 
                #     #         self.ave_v * self.T, 
                #     #         0
                #     #     )
                #     # )
                #     ax.add_artist(plt.Circle((self.state.x, self.state.y), self.working_width / 2, color=color, alpha=0.4, linewidth=0))
                # else:
                ax.add_artist(plt.Circle((self.state.x, self.state.y), self.working_width*0.28, color=color, alpha=1, linewidth=0))
            if return_bound:
                return (f1, f2), car_shape[:4]
        elif mode == 2:
            car_shape = np.array([
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
                [self.state.x+np.sin(self.state.psi)*self.K / 2 , self.state.y-np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L - np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L + np.cos(self.state.psi)*self.K / 2],
                [self.state.x + np.cos(self.state.psi)*self.L + np.sin(self.state.psi)*self.K / 2 , self.state.y + np.sin(self.state.psi)*self.L - np.cos(self.state.psi)*self.K / 2],
                [self.state.x-np.sin(self.state.psi)*self.K / 2 , self.state.y+np.cos(self.state.psi)*self.K / 2],
            ])
            if show:
                f1 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
                f2 = ax.plot(car_shape[:,0],car_shape[:,1], color = color)
                if self.working:
                    # ax.add_patch(
                    #     plt.Rectangle(
                    #         (car_shape[0, 0], car_shape[0, 1]), 
                    #         self.K, 
                    #         self.ave_v * self.T, 
                    #         0
                    #     )
                    # )
                    ax.add_artist(plt.Circle((self.state.x, self.state.y), 4))
                else:
                    ax.add_artist(plt.Circle((self.state.x, self.state.y), 2))
            if return_bound:
                return (f1, f2), car_shape[:4]
        else:
            if show:
                f1 = ax.plot(self.state.x, self.state.y, marker = '.', color = color)
            return (f1, f2)


