from algo.policy_gradiant import PG
from algo.ppo import PPO
from algo.npg import NPG
from algo.trpo import TRPO
from algo.ppo_lagrangian import PPO_Lag
from algo.esb_ppo_lagrangian import ESB_PPO_Lag
from algo.esb_ppo_lagrangian_g1 import ESB_PPO_Lag_G1
from algo.esb_ppo_lagrangian_g2 import ESB_PPO_Lag_G2
from algo.esb_ppo_lagrangian_adv import ESB_PPO_Lag_ADV
from algo.static_arrange import MGGA
from algo.dynamic_arrange import IIG
from model.ac import GNNAC

REGISTRY = {
    'pg': PG,
    'ppo': PPO,
    'npg': NPG,
    'trpo': TRPO,
    'ppo_lag': PPO_Lag,
    'esb_ppo_lag': ESB_PPO_Lag,
    'esb_ppo_lag_g1': ESB_PPO_Lag_G1,
    'esb_ppo_lag_g2': ESB_PPO_Lag_G2,
    'esb_ppo_lag_adv': ESB_PPO_Lag_ADV,
}
