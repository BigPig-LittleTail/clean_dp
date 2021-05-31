import os
from tqdm import tqdm

# for i in tqdm(range(10), desc="4-2 idec+lt"):
#     os.system("python train_idec+lt.py \
#         --model_type aidec \
#         --x_path weibo/embedding/3-1/self_other_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/aae/idec+lt-sdcn+lt.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 \
#         --alpha 0.1 \
#         --beta 0.0 >> result/4-2/idec+lt.txt")
#
#
# for i in tqdm(range(10), desc="4-2 sdcn+lt"):
#     os.system("python train_sdcn+lt.py \
#         --model_type tsdcn \
#         --x_path weibo/embedding/3-1/self_other_enhance \
#         --y_path weibo/label \
#         --pre_path weibo/pretrain/aae/idec+lt-sdcn+lt.pkl \
#         --train_epoch 800 \
#         --n_cluster 30 \
#         --alpha 0.1 \
#         --beta 0.0 >> result/4-2/sdcn+lt.txt")


for i in tqdm(range(10), desc="4-2 daegc+lt"):
    os.system("python train_daegc+lt.py \
    --x_path weibo/embedding/3-1/self_other_enhance \
    --y_path weibo/label \
    --pre_path weibo/pretrain/agae/4-1/daegc+lt.pkl \
    --n_z 50 \
    --beta 10 \
    --gamma 0.0 >> result/4-2/daegc+lt.txt")