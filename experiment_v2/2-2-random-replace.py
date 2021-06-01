import os
from tqdm import tqdm

# for i in tqdm(range(10), desc="2-2 idec+random"):
#     os.system("python train_idec.py \
#     --model_type idec \
#     --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity \
#     --y_path weibo/label \
#     --pre_path experiment_v2/weibo/pretrain/ae/2-2/random_replace_200.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> experiment_v2/result_v2/2-2/idec+random.txt")


# for i in tqdm(range(10), desc="2-2 sdcn+random"):
#     os.system("python train_sdcn.py \
#     --model_type sdcn \
#     --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity \
#     --y_path weibo/label \
#     --pre_path experiment_v2/weibo/pretrain/ae/2-2/random_replace_30.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> experiment_v2/result_v2/2-2/sdcn+random.txt")


# for i in tqdm(range(10), desc="2-2 daegc+random"):
#     os.system("python train_daegc.py \
#         --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity \
#         --y_path weibo/label \
#         --pre_path experiment_v2/weibo/pretrain/gae/2-2/random_replace_30.pkl \
#         --n_z 50 \
#         --beta 10 >> experiment_v2/result_v2/2-2/daegc+random.txt ")



os.system("python kmeans.py \
--x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity_new \
--y_path weibo/label \
--n_cluster 30 >> experiment_v2/result_v2/2-2/kmeans_new.txt")


# for i in tqdm(range(10), desc="2-2 idec+random"):
#     os.system("python train_idec.py \
#     --model_type idec \
#     --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity_new \
#     --y_path weibo/label \
#     --pre_path experiment_v2/weibo/pretrain/ae/2-2/idec_random_replace_200.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> experiment_v2/result_v2/2-2/idec+random_new.txt")
#
#
# for i in tqdm(range(10), desc="2-2 sdcn+random"):
#     os.system("python train_sdcn.py \
#     --model_type sdcn \
#     --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity_new \
#     --y_path weibo/label \
#     --pre_path experiment_v2/weibo/pretrain/ae/2-2/sdcn_random_replace_30.pkl \
#     --train_epoch 800 \
#     --n_cluster 30 >> experiment_v2/result_v2/2-2/sdcn+random_new.txt")


# for i in tqdm(range(10), desc="2-2 daegc+random"):
#     os.system("python train_daegc.py \
#         --x_path experiment_v2/weibo/embedding/2-2/random_replace_num_with_entity_new \
#         --y_path weibo/label \
#         --pre_path experiment_v2/weibo/pretrain/gae/2-2/daegc_random_replace_30.pkl \
#         --n_z 50 \
#         --beta 10 >> experiment_v2/result_v2/2-2/daegc+random_new.txt ")
