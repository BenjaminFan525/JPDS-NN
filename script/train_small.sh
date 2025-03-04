/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/algo/train_test.py \
        --seed 27688426 \
        --train_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_md/Train \
                /home/fanyx/mdvrp/data/Gdataset/Task_1end/Train \
                /home/fanyx/mdvrp/data/Gdataset/Task_1depot/Train \
        --valid_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_md/Valid \
                /home/fanyx/mdvrp/data/Gdataset/Task_1end/Valid \
                /home/fanyx/mdvrp/data/Gdataset/Task_1depot/Valid \
        --test_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_md/Test \
                /home/fanyx/mdvrp/data/Gdataset/Task_1end/Test \
                /home/fanyx/mdvrp/data/Gdataset/Task_1depot/Test \
        --log_dir \
        /home/fanyx/mdvrp/result/training_rl \
        --config \
        /home/fanyx/mdvrp/config/common.yaml \
        --epochs 40 \
        --device cuda:1 \
        --rl_algo ppo \
        --obj t \
        --fusing_s \
        --fusing_epoch 7 \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2025-02-17__21-18/best_model15.pt \
        # --total_time \
        # --total_time_epoch 6
        # --veh_reciprocal
        
        
