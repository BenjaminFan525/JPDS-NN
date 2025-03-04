/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/experiment/Test_compare/test_compare_ga.py \
        --test_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/multi_depot \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_end \
        --save_dir experiment/Test_compare \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2025-02-07__06-10__s/best_model34.pt \
        --obj s \
        --batch_size 1 \
        --fig_interval 100000 \
        --num_workers 75 \