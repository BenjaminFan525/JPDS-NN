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
        --epochs 40 \
        --device cuda:0 \
        --rl_algo ppo \
        --obj t \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2024-12-25__22-46/best_model10.pt \
        --fusing_s \
        --fusing_epoch 7 \
        # --total_time \
        # --total_time_epoch 6
        # --veh_reciprocal
        
        
