python /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/train_test.py \
        --seed 12345 \
        --train_data \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/1_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/1_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/1_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/2_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/2_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/2_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/3_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/3_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/3_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/4_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/4_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/4_7 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/6_3 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/6_5 \
        /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Task_Train_xs/6_7 \
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
        --rl_algo trpo \
        --obj t