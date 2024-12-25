/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/algo/train_test.py \
        --seed 4279149 \
        --train_data \
        /home/fanyx/mdvrp/data/Gdataset/Task_small/Train \
        --valid_data \
        /home/fanyx/mdvrp/data/Gdataset/Task_small/Validation \
        --test_data \
        /home/fanyx/mdvrp/data/Gdataset/Task_small/Test \
        --log_dir \
        /home/fanyx/mdvrp/result/training_rl \
        --config \
        /home/fanyx/mdvrp/config/common.yaml \
        --epochs 25 \
        --device cuda:1 \
        --rl_algo ppo \
        --obj t \
        --fusing_s \
        --fusing_epoch 4 \
        # --total_time \
        # --total_time_epoch 6
        # --veh_reciprocal
        
        # --checkpoint /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-02-15__23-48_base/best_model18.pt
