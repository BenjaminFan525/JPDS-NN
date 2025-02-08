/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/experiment/dynamic_arrangement/rearrange_ga.py \
        --test_data \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/multi_depot \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_depot \
                /home/fanyx/mdvrp/data/Gdataset/Task_test_multifield/single_end \
        --save_dir /home/fanyx/mdvrp/experiment/dynamic_arrangement \
        --checkpoint /home/fanyx/mdvrp/result/training_rl/ppo/2025-02-02__08-36__t/best_model38.pt \
        --batch_size 1 \
        --stop_coeff 0.5 \
        --fig_interval 1000 \
        --num_workers 25 \