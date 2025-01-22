/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/algo/train_test.py \
        --seed 16874536 \
        --train_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_md/Train \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1end/Train \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1depot/Train \
        --valid_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_md/Valid \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1end/Valid \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1depot/Valid \
        --test_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_md/Test \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1end/Test \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_1depot/Test \
        --log_dir \
        /home/fanyx/mdvrp/result/training_rl \
        --config \
        /home/fanyx/mdvrp/config/common.yaml \
        --epochs 60 \
        --device cuda:1 \
        --rl_algo ppo \
        --obj t \
        --fusing_s \
        --fusing_epoch 10 \
        # --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2024-12-25__22-46/best_model10.pt \
        # --total_time \
        # --total_time_epoch 6
        # --veh_reciprocal
        
        
