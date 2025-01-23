/home/fanyx/anaconda3/envs/ia/bin/python /home/fanyx/mdvrp/algo/GNN_pre_train_ia.py \
        --seed 4594259 \
        --train_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Train/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Train/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Train/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Train/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Train/6 \
        --valid_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Validation/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Validation/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Validation/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Validation/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Validation/6 \
        --test_data \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Test/1 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Test/2 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Test/3 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Test/4 \
            /home/fanyx/mdvrp/data/Gdataset/GNN_ia/Test/6 \
        --log_dir /home/fanyx/mdvrp/result/GNN_pretrain \
        --config /home/fanyx/mdvrp/config/gnn_common.yaml \
        --epochs 600 \
        --device cuda:0 \
        --batch_size 32