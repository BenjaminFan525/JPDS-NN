python /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/GNN_pre_train.py \
        --seed 12345 \
        --train_data \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Train/1 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Train/2 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Train/3 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Train/4 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Train/6 \
        --valid_data \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Validation/6 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Validation/1 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Validation/2 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Validation/3 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Validation/4 \
        --test_data \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/1 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/2 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/3 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/4 \
            /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/6 \
        --log_dir /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs \
        --config /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/configs/gnn_common.yaml \
        --checkpoint /home/xuht/Intelligent-Agriculture/ia/algo/arrangement/runs/2024-01-07__14-36/best_model785.pt \
        --test \
        --epochs 5000 \
        --device cuda:1 \
        --batch_size 4
        # /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/1 \
        #     /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/2 \
        #     /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/3 \
        #     /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/4 \
        #     /home/xuht/Intelligent-Agriculture/ia/Dataset/Gdataset/Test/6 \