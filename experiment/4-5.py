import os
from tqdm import tqdm

for i in tqdm(range(10), desc="4-5 idec"):
    os.system("python train_idec.py \
    --model_type idec \
    --x_path stackoverflow/embedding \
    --y_path stackoverflow/label \
    --pre_path stackoverflow/200_new.pkl \
    --train_epoch 200 \
    --n_cluster 20 >> result/4-5/idec_new.txt")



for i in tqdm(range(10), desc="4-5 idec+lt"):
    os.system("python train_idec+lt.py \
        --model_type aidec \
        --x_path stackoverflow/embedding \
        --y_path stackoverflow/label \
        --pre_path stackoverflow/200+lt+0.001.pkl \
        --train_epoch 200 \
        --n_cluster 20 \
        --alpha 0.1 \
        --beta 0.1 >> result/4-5/idec+lt+0.001.txt")