/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/experiment/test_compare_rl.py \
        --seed 4279149 \
        --batch_size 10 \
        --test_data \
        /home/fanyx/mdvrp/data/Gdataset/Task_test_zz \
        --log_dir \
        /home/fanyx/mdvrp/experiment/Test_compare \
        --config \
        /home/fanyx/mdvrp/config/common.yaml \
        --device cuda:0 \
        --rl_algo ppo \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2024-12-27__12-11/best_model31.pt \
        --checkpoint_ia /home/fanyx/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt 