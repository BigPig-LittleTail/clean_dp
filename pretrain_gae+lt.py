import argparse

import torch
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import GAE
from torch_geometric.utils import to_undirected

from model.model import ADAEGCEncoder

from pretrain_ae import nmf
from model.utils import construct_edge_index_direction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--x_path", type=str, default="")
    parser.add_argument("--y_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pre_epoch", type=int, default=30)
    parser.add_argument("--input_dim", type=int, default=2000)
    parser.add_argument("--n_z", type=int, default=10)
    parser.add_argument("--n_cluster", type=int, default=30)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=10)

    args = parser.parse_args()

    lr = args.lr
    pre_epoch = args.pre_epoch
    input_dim = args.input_dim
    n_z = args.n_z
    n_cluster = args.n_cluster

    seed = args.seed
    alpha = args.alpha
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    x_path = args.x_path
    y_path = args.y_path
    save_path = args.save_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int32)

    x_topic = nmf(feature, n_cluster, seed=seed)
    x_topic = torch.from_numpy(x_topic).to(device)
    encoder = ADAEGCEncoder(4, 1, 500, input_dim, n_z, n_cluster,
                            att_dropout=0.0, fdd_dropout=0.0)

    x = torch.from_numpy(feature)

    edge_index_direction = torch.from_numpy(construct_edge_index_direction(x)).t().contiguous()
    edge_index = to_undirected(edge_index_direction, num_nodes=x.shape[0])

    model = GAE(encoder).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for i in range(pre_epoch):
        z, q, c = model.encode(x, edge_index)
        c_loss = F.kl_div(c, x_topic, reduction='batchmean')

        out_edge = model.decode(z, edge_index)
        re_loss = F.mse_loss(out_edge, torch.ones_like(out_edge, dtype=torch.float32))

        loss = re_loss + alpha * c_loss
        print("re_loss:{}, c_loss:{}".format(re_loss, c_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_path)



