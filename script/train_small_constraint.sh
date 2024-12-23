python /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/train_test.py \
        --seed 6870 \
        --train_data \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/2_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/2_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/2_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/3_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/3_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/3_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/4_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/4_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_small/4_7 \
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
        /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_safe_rl \
        --config \
        /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/configs/common_constraint.yaml \
        --epochs 25 \
        --device cuda:1 \
        --rl_algo esb_ppo_lag_g2 \
        --obj c \
        --constraint t \
        --fusing_s \
        --fusing_epoch 25 \
        # --recount_epoch \
        # --checkpoint /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-04__12-53_t/best_model39.pt
        # --checkpoint /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs_rl/ppo/2024-03-05__11-05_c/best_model23.pt
        
        # --veh_reciprocal
        # --total_time \
        # --total_time_epoch 5
        
