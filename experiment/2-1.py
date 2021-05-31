import os
from tqdm import tqdm

for i in tqdm(range(10), desc="2-1 idec"):
    os.system("python train_idec.py \
    --model_type idec \
    --x_path reut/reduce_reut.txt \
    --y_path reut/reut_label.txt \
    --pre_path reut/reduce_reut_30_001.pkl \
    --train_epoch 200 \
    --n_cluster 4 >> result/2-1/idec.txt")


for i in tqdm(range(10), desc="2-1 sdcn"):
    os.system("python train_sdcn.py \
    --model_type sdcn \
    --x_path reut/reduce_reut.txt \
    --y_path reut/reut_label.txt \
    --pre_path reut/reduce_reut_30_001.pkl \
    --train_epoch 200 \
    --n_cluster 4 >> result/2-1/sdcn.txt")