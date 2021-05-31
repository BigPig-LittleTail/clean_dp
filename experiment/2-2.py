import os
from tqdm import tqdm

# for i in tqdm(range(10), desc="2-2 idec"):
#     os.system("python train_idec.py \
#     --model_type idec \
#     --x_path weibo/embedding/2-2/no_ent_embedding \
#     --y_path weibo/label \
#     --pre_path weibo/pretrain/ae/2-2/200.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> result/2-2/idec.txt")
#
#
# for i in tqdm(range(10), desc="2-2 sdcn"):
#     os.system("python train_sdcn.py \
#     --model_type sdcn \
#     --x_path weibo/embedding/2-2/no_ent_embedding \
#     --y_path weibo/label \
#     --pre_path weibo/pretrain/ae/2-2/30.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> result/2-2/sdcn.txt")

for i in tqdm(range(10), desc="2-2 daegc"):
    os.system("python train_daegc.py \
        --x_path weibo/embedding/2-2/no_ent_embedding \
        --y_path weibo/label \
        --pre_path weibo/pretrain/gae/2-2/30.pkl \
        --n_z 50 \
        --beta 10 >> result/2-2/daegc.txt ")
