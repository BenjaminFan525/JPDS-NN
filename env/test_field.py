import env.dubins as dubins
from shapely import geometry
import geopandas
import utm
from geopandas import GeoSeries
import numpy as np
import matplotlib.pyplot as plt

import yaml

from env.field import Field
from algo import MGGA
import utils.common as ucommon

def path_gen(path, R = 4.8, width = 24, ax = None, color = 'r', name = "path"):
    target = np.zeros((len(path), 3))
    lat_lon_path = []
    for idx, point in enumerate(path):
        target[idx][0], target[idx][1], ran, num = utm.from_latlon(point[0], point[1])
        if idx % 2 == 1:
            dir = np.array(target[idx][:2]) - np.array(target[idx - 1][:2])
            # print(path[idx])
            # print(path[idx - 1])
            lenth = np.linalg.norm(dir)
            # print(lenth)
            dir /= lenth
            angle = np.arctan2(dir[1], dir[0]).item()
            target[idx][2] = angle
            target[idx - 1][2] = angle
            if ax:
                n_dir = np.array([[0, -1],[1, 0]]) @ dir
                xy = target[idx][:2] + n_dir * width / 2
                rect = plt.Rectangle((xy[0], xy[1]), lenth, width, angle=np.rad2deg(angle) + 180, alpha=0.5, color=color, fill=True)
                ax.add_patch(rect)
    
    path_g = dubins.Dubins(R, 2)
    real_path = path_g.dubins_multi(target)[0]

    for point in real_path:
        lat_lon_path.append(utm.to_latlon(point[0], point[1], ran, num))
        with open(name + '.txt', 'a') as f:
            for p in lat_lon_path:
                f.write(str(p[0]) + ', ' + str(p[1]) + '\r\n')
    return real_path

# def render(path, width):
#     car_shape = np.array([
#                 [path[0][0]-np.sin(path[0][2]) * self.K / 2 , self.state.y+np.cos(path[0][2])*self.K / 2],
#                 [path[0][0]+np.sin(path[0][2]) * self.K / 2 , self.state.y-np.cos(path[0][2])*self.K / 2],
#                 [path[0][0] + np.cos(path[0][2]) * self.L + np.sin(path[0][2])*self.K / 2 , self.state.y + np.sin(path[0][2])*self.L - np.cos(path[0][2])*self.K / 2],
#                 [path[0][0] + np.cos(path[0][2]) * self.L - np.sin(path[0][2])*self.K / 2 , self.state.y + np.sin(path[0][2])*self.L + np.cos(path[0][2])*self.K / 2],
#                 [path[0][0]-np.sin(path[0][2]) * self.K / 2 , self.state.y+np.cos(path[0][2])*self.K / 2],
#             ])

path1 = [
    # [47.5898917735, 131.992345902], #A
    # [47.5886885455, 131.991224189], #B
    # [47.5886309834, 131.991359103], #B
    # [47.5898342101, 131.992480818], #A
    [47.58986299328, 131.99241336496],
    [47.58865976596, 131.99129165108],
    [47.58860220381, 131.99142656513],
    [47.58980542978, 131.99254828076],
]
path2 = [
    [47.5897190828, 131.992750649],
    [47.5885158588, 131.991628931],
    # [47.5883431708, 131.99203367],
    # [47.5895463907, 131.993155393],
    [47.5884007336, 131.991898757],
    [47.5896039548, 131.993020479],
]
path3 = [
    [47.5894888263, 131.993290307], #A
    [47.5882856078, 131.992168582], #B
    [47.5881704813, 131.992438406],
    [47.5893736972, 131.993560135],
    # [47.5881129179, 131.992573317], #B
    # [47.5893161324, 131.993695048], #A
    # [47.5879402266, 131.99297805], #B
    # [47.5891434371, 131.994099786], #A

]


path4 = [
    # [47.58993494744, 131.99224471980],
    # [47.58873171843, 131.99112300809],
    # [47.58858781322, 131.99146029339],
    # [47.58979103890, 131.99258200971],
    [47.58990616728, 131.99231218288],
    [47.58870293895, 131.99119047028],
    [47.58870919822, 131.99110093320],
    [47.58809006957, 131.99255203232],
    [47.58804443560, 131.99250949394],
    [47.58866356370, 131.99105839552],
    [47.58873171843, 131.99112300809],
    [47.58993494744, 131.99224471980],
]

path5 = [
    # [47.58991459266, 131.99236717946],
    
    # [47.58996022949, 131.99240972947],
    [47.58995809503, 131.99226520729],
    [47.58933895130, 131.99371632580],
    [47.58930174559, 131.99372879100],
    [47.58809853145, 131.99260706013],
    [47.58806975112, 131.99267452069],
    [47.58927296459, 131.99379625244],
    [47.58938458761, 131.99375887649],
    [47.59000438724, 131.99230617243],
]

print(path4[2][0]-path1[2][0])
print(path4[2][1]-path1[2][1])

_, ax = plt.subplots()
ax.axis('equal')

path1_ = path_gen(path1, width=12, color='r', name='path1')
path2_ = path_gen(path2, width=24, color='g', name='path2')
path3_ = path_gen(path3, width=24, color='b', name='path3')
path4_ = path_gen(path4, 2, width=6, color='c', name='path4')
path5_ = path_gen(path5, 2, width=6, color='m', name='path5')

bound = np.array([
    [ 724960.24840212, 5275080.66318828],
    [ 725082.75197389, 5275010.61610567],
    [ 724990.33446884, 5274849.89352426],
    [ 724866.07019765, 5274917.1912237 ],
    [ 724960.24840212, 5275080.66318828],
])

# ax.plot(path1_[:, 0], path1_[:, 1], 'r', linewidth = 12, alpha = 0.5)
ax.plot(path1_[:, 0], path1_[:, 1], '--r', label='永佳无人拖拉机')
# ax.plot(path2_[:, 0], path2_[:, 1], 'g', linewidth = 24, alpha = 0.5)
ax.plot(path2_[:, 0], path2_[:, 1], '--g', label='Case无人拖拉机')
# ax.plot(path3_[:, 0], path3_[:, 1], 'b', linewidth = 24, alpha = 0.5)
ax.plot(path3_[:, 0], path3_[:, 1], '--b', label='东方红无人拖拉机')
ax.plot(path4_[:, 0], path4_[:, 1], '--c', label='植保无人机1号')
ax.plot(path5_[:, 0], path5_[:, 1], '--m', label='植保无人机2号')
plt.legend()
# ax.plot(bound[:, 0], bound[:, 1], 'k')
plt.show()