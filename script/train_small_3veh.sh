python /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/train_test.py \
        --seed 12345 \
        --train_data \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small_3veh/2_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small_3veh/3_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small_3veh/4_3 \
        --valid_data \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/1_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/1_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/1_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/2_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/2_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/2_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/3_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/3_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/3_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/4_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/4_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/4_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/6_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/6_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Validation/6_7 \
        --test_data \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/1_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/1_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/1_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/2_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/2_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/2_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/3_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/3_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/3_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/4_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/4_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/4_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/6_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/6_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Test/6_7 \
        --log_dir \
        /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl \
        --config \
        /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/configs/common.yaml \
        --epochs 20 \
        --device cuda:1 \
        --rl_algo ppo \
        --obj t \
        --checkpoint /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-02-23__16-13_s/best_model13.pt \
        --recount_epoch \
        # --fusing_s \
        # --fusing_epoch 5 \
        # --veh_reciprocal
        # --total_time \
        # --total_time_epoch 5
        
