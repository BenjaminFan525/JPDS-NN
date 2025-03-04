/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/experiment/Test_compare/test_compare_ga_large.py \
        --test_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_large \
        --save_dir experiment/Test_compare \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt \
        --obj t \
        --batch_size 1 \
        --fig_interval 100000 \
        --num_workers 40 \