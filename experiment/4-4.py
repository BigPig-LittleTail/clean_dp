import os
from tqdm import tqdm

for i in tqdm(range(10), desc="4-4 idec"):
    os.system("python train_idec.py \
    --model_type idec \
    --x_path THUCNews/2000.txt \
    --y_path THUCNews/label \
    --pre_path THUCNews/200.pkl \
    --train_epoch 200 \
    --n_cluster 4 >> result/4-4/idec.txt")


for i in tqdm(range(10), desc="4-4 sdcn"):
    os.system("python train_sdcn.py \
    --model_type sdcn \
    --x_path THUCNews/2000.txt \
    --y_path THUCNews/label \
    --pre_path THUCNews/30.pkl \
    --train_epoch 200 \
    --n_cluster 4 >> result/4-4/sdcn.txt")


for i in tqdm(range(10), desc="4-4 daegc"):
    os.system("python train_daegc.py \
    --x_path THUCNews/2000.txt \
    --y_path THUCNews/label \
    --pre_path THUCNews/gae_30.pkl \
    --n_z 50 \
    --n_cluster 4 \
    --beta 10 >> result/4-4/daegc.txt ")



for i in tqdm(range(10), desc="4-4 idec+lt"):
    os.system("python train_idec+lt.py \
        --model_type aidec \
        --x_path THUCNews/2000.txt \
        --y_path THUCNews/label \
        --pre_path THUCNews/200+lt.pkl\
        --train_epoch 200 \
        --n_cluster 4 \
        --alpha 0.1 \
        --beta 0.1 >> result/4-4/idec+lt.txt")


for i in tqdm(range(10), desc="4-4 sdcn+lt"):
    os.system("python train_sdcn+lt.py \
        --model_type tsdcn \
        --x_path THUCNews/2000.txt \
        --y_path THUCNews/label \
        --pre_path THUCNews/200+lt.pkl\
        --train_epoch 200 \
        --n_cluster 4 \
        --alpha 0.1 \
        --beta 0.1 >> result/4-4/sdcn+lt.txt")


for i in tqdm(range(10), desc="4-4 daegc+lt"):
    os.system("python train_daegc+lt.py \
    --x_path THUCNews/2000.txt \
    --y_path THUCNews/label \
    --pre_path THUCNews/gae_30+lt.pkl \
    --n_z 50 \
    --n_cluster 4 \
    --beta 10 >> result/4-4/dagec+lt.txt")
