/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/algo/GNN_pre_train.py \
        --seed 4594259 \
        --train_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Train/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Train/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Train/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Train/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Train/6 \
        --valid_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Validation/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Validation/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Validation/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Validation/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Validation/6 \
        --test_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Test/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Test/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Test/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Test/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_pure/Test/6 \
        --log_dir /home/fanyx/mdvrp/result/GNN_pretrain \
        --config /home/fanyx/mdvrp/config/gnn_common.yaml \
        --epochs 1000 \
        --device cuda:2 \
        --batch_size 64