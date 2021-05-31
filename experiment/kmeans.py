import os
os.system("python kmeans.py --x_path weibo/embedding/3-1/process_embed \
--y_path weibo/label \
--n_cluster 30 >> result/kmeans/r.txt")

os.system("python kmeans.py --x_path weibo/embedding/3-1/self_enhance \
--y_path weibo/label \
--n_cluster 30 >> result/kmeans/s.txt")

os.system("python kmeans.py --x_path weibo/embedding/3-1/self_other_enhance \
--y_path weibo/label \
--n_cluster 30 >> result/kmeans/s_o.txt")

os.system("python kmeans.py \
--x_path weibo/embedding/2-2/no_ent_embedding \
--y_path weibo/label \
--n_cluster 30 >> result/kmeans/no_entity.txt")

os.system("python kmeans.py \
--x_path reut/reduce_reut.txt \
--y_path reut/reut_label.txt \
--n_cluster 4 >> result/kmeans/reut_reduce.txt")