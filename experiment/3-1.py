import os
from tqdm import tqdm

# for i in tqdm(range(10), desc="3-1 idec_r"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-1/process_embed \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/idec_r.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/idec_r.txt")
#
# for i in tqdm(range(10), desc="3-1 idec_s"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-1/self_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/idec_s.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/idec_s.txt ")
#
# for i in tqdm(range(10), desc="3-1 idec_s_o"):
#     os.system("python train_idec.py \
#         --model_type idec \
#         --x_path weibo/embedding/3-1/self_other_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/idec_s_o.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/idec_s_o.txt ")
#
# for i in tqdm(range(10), desc="3-1 sdcn_r"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-1/process_embed \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/sdcn_r.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/sdcn_r.txt")
#
# for i in tqdm(range(10), desc="3-1 sdcn_s"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-1/self_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/sdcn_s.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/sdcn_s.txt")
#
# for i in tqdm(range(10), desc="3-1 sdcn_s_o"):
#     os.system("python train_sdcn.py \
#         --model_type sdcn \
#         --x_path weibo/embedding/3-1/self_other_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/ae/3-1/sdcn_s_o.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 >> result/3-1/sdcn_s_o.txt")

for i in tqdm(range(10), desc="3-1 daegc_r"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-1/process_embed \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-1/daegc_r.pkl \
    --n_z 50 \
    --beta 10 >> result/3-1/daegc_r.txt")

for i in tqdm(range(10), desc="3-1 daegc_s"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-1/process_embed \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-1/daegc_s.pkl \
    --n_z 50 \
    --beta 10 >> result/3-1/daegc_s.txt")


for i in tqdm(range(10), desc="3-1 daegc_s_o"):
    os.system("python train_daegc.py \
    --x_path weibo/embedding/3-1/self_other_enhance \
    --y_path weibo/label \
    --pre_path weibo/pretrain/gae/3-1/daegc_s_o.pkl \
    --n_z 50 \
    --beta 10 >> result/3-1/daegc_s_o.txt")

