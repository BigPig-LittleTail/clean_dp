import argparse

import torch
import numpy as np
from sklearn.cluster import KMeans
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import GAE
from torch_geometric.utils import to_undirected

from model import AE, TopicDataSet, ADAEGCEncoder, DAEGCEncoder

from utils import construct_edge_index_direction, eva, target_distribution



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--x_path", type=str, default="")
    parser.add_argument("--y_path", type=str, default="")
    parser.add_argument("--pre_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train_epoch", type=int, default=200)
    parser.add_argument("--input_dim", type=int, default=2000)
    parser.add_argument("--n_z", type=int, default=10)
    parser.add_argument("--n_cluster", type=int, default=30)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--beta", type=float, default=10)

    args = parser.parse_args()

    lr = args.lr
    train_epoch = args.train_epoch
    input_dim = args.input_dim
    n_z = args.n_z
    n_cluster = args.n_cluster

    seed = args.seed
    beta = args.beta
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    x_path = args.x_path
    y_path = args.y_path
    pre_path = args.pre_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int32)

    encoder = DAEGCEncoder(4, 1, 500, input_dim, n_z, n_cluster,
                       att_dropout=0.0, fdd_dropout=0.0)

    x = torch.from_numpy(feature)

    edge_index_direction = torch.from_numpy(construct_edge_index_direction(x)).t().contiguous()
    edge_index = to_undirected(edge_index_direction, num_nodes=x.shape[0])

    model = GAE(encoder)
    model.load_state_dict(torch.load(pre_path))

    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    optimizer = Adam(model.parameters(), lr=lr)


    with torch.no_grad():
        z, q = model.encode(x, edge_index)

    kmeans = KMeans(n_clusters=n_cluster, n_init=20, random_state=seed)
    k_predict = kmeans.fit_predict(z.data.cpu().numpy())
    encoder.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    eva(y, k_predict, "pretrain")

    for epoch in range(train_epoch):
        z, q = model.encode(x, edge_index)

        out_edge = model.decode(z, edge_index)

        if epoch % 1 == 0:
            p = target_distribution(q.data)
            res1 = q.data.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P
            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'P')

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(out_edge, torch.ones_like(out_edge))

        loss = re_loss + beta * kl_loss
        print("kl_loss{}, re_loss{}".format(kl_loss, re_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

