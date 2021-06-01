import os
from tqdm import tqdm


for i in tqdm(range(10), desc="2-2 sdcn+original_knn.txt"):
    os.system("python experiment_v2/train_sdcn.py \
    --model_type sdcn \
    --x_path weibo/embedding/2-2/no_ent_embedding \
    --y_path weibo/label \
    --graph_x_path weibo/embedding/3-1/process_embed \
    --pre_path weibo/pretrain/ae/2-2/30.pkl \
    --train_epoch 800 \
    --n_cluster 30 >> experiment_v2/result_v2/2-2/sdcn+original_knn.txt")


for i in tqdm(range(10), desc="2-2 sdcn+original_embedding.txt"):
    os.system("python experiment_v2/train_sdcn.py \
    --model_type sdcn \
    --x_path weibo/embedding/3-1/process_embed \
    --y_path weibo/label \
    --graph_x_path weibo/embedding/2-2/no_ent_embedding \
    --pre_path weibo/pretrain/ae/3-1/sdcn_r.pkl \
    --train_epoch 800 \
    --n_cluster 30 >> experiment_v2/result_v2/2-2/sdcn+original_embedding.txt")


for i in tqdm(range(10), desc="2-2 daegc+original_knn.txt"):
    os.system("python experiment_v2/train_daegc.py \
        --x_path weibo/embedding/2-2/no_ent_embedding \
        --y_path weibo/label \
        --graph_x_path weibo/embedding/3-1/process_embed \
        --pre_path experiment_v2/weibo/pretrain/gae/2-2/original_knn_no_entity_embedding.pkl \
        --n_z 50 \
        --beta 10 >> experiment_v2/result_v2/2-2/daegc+original_knn.txt ")


for i in tqdm(range(10), desc="2-2 daegc+original_embedding.txt"):
    os.system("python experiment_v2/train_daegc.py \
        --x_path weibo/embedding/3-1/process_embed \
        --y_path weibo/label \
        --graph_x_path weibo/embedding/2-2/no_ent_embedding \
        --pre_path experiment_v2/weibo/pretrain/gae/2-2/original_embedding_no_entity_knn.pkl \
        --n_z 50 \
        --beta 10 >> experiment_v2/result_v2/2-2/daegc+original_embedding.txt ")


