from env.car import car
from env.car import DRIVE_MODE
import matplotlib.pyplot as plt
import numpy as np

car1=car("car1.yaml")
des=np.pi/4
car1.v_des=2
car1.psi_des=des
psi=[]
th=[]
th_=[]
th_d=[]
plt.plot(car1.state.x,car1.state.y,'.')
t=400
for i in range(t):
    car1.update_state(DRIVE_MODE.AUTO_VEL_PSI)
    psi.append(car1.state.psi)
    th.append(car1.theta)
    th_.append(car1.theta_)
    # th_d.append(np.arctan(car1.w_des*car1.L/car1.state.v))
    plt.plot(car1.state.x,car1.state.y,'.')

plt.figure(2)
plt.plot(des*np.ones(t),'--')
plt.plot(psi)
plt.plot(th)
plt.plot(th_)
# plt.plot(th_d,'--')
plt.legend(["w_des","w","th","th_"])
plt.ylim([-2,3])

plt.show()
    