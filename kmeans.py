import argparse
import numpy as np
from sklearn.cluster import KMeans

from model.utils import eva

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--x_path", type=str, default="")
    parser.add_argument("--y_path", type=str, default="")
    parser.add_argument("--n_cluster", type=int)

    args = parser.parse_args()

    n_cluster = args.n_cluster

    x_path = args.x_path
    y_path = args.y_path

    x = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int32)
    
    for i in range(10):
        k_t = KMeans(n_clusters=n_cluster, n_init=20)
        k_t_pre = k_t.fit_predict(x)
        eva(y, k_t_pre, 'kmeans')