/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/experiment/Test_compare/test_compare_rl.py \
        --seed 4279149 \
        --batch_size 10 \
        --test_data \
        /home/fanyx/mdvrp/data/Gdataset/Task_test_1depot \
        --log_dir \
        /home/fanyx/mdvrp/experiment/Test_compare \
        --config \
        /home/fanyx/mdvrp/config/common.yaml \
        --device cuda:0 \
        --rl_algo ppo \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2025-01-20__11-03__large_md/best_model32.pt \
        --checkpoint_ia /home/fanyx/mdvrp/result/training_rl/ppo/2025-01-23__19-04__ia/best_model37.pt \