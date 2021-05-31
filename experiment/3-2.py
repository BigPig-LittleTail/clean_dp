import os
from tqdm import tqdm

# for i in tqdm(range(10), desc="3-2 5_idec"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-2/5 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/5_idec.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/5_idec.txt ")
#
#
# for i in tqdm(range(10), desc="3-2 15_idec"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-2/15 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/15_idec.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/15_idec.txt ")
#
# for i in tqdm(range(10), desc="3-2 20_idec"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-2/20 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/20_idec.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/20_idec.txt ")
#
# for i in tqdm(range(10), desc="3-2 5_sdcn"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-2/5 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/5_sdcn.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/5_sdcn.txt ")
#
# for i in tqdm(range(10), desc="3-2 15_sdcn"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-2/15 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/15_sdcn.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/15_sdcn.txt ")
#
# for i in tqdm(range(10), desc="3-2 20_sdcn"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-2/20 \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-2/20_sdcn.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-2/20_sdcn.txt ")
    
for i in tqdm(range(10), desc="3-2 5_daegc"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-2/5 \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-2/5.pkl \
    --n_z 50 \
    --beta 10 >> result/3-2/5_daegc.txt")
    

for i in tqdm(range(10), desc="3-2 15_daegc"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-2/15 \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-2/15.pkl \
    --n_z 50 \
    --beta 10 >> result/3-2/15_daegc.txt")


for i in tqdm(range(10), desc="3-2 20_daegc"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-2/20 \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-2/20.pkl \
    --n_z 50 \
    --beta 10 >> result/3-2/20_daegc.txt")

