import argparse

import torch
import os
import sys
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import GAE
from torch_geometric.utils import to_undirected

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.model import DAEGCEncoder

from model.utils import construct_edge_index_direction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--x_path", type=str, default="")
    parser.add_argument("--y_path", type=str, default="")
    parser.add_argument("--graph_x_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pre_epoch", type=int, default=30)
    parser.add_argument("--input_dim", type=int, default=2000)
    parser.add_argument("--n_z", type=int, default=10)
    parser.add_argument("--n_cluster", type=int, default=30)

    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    lr = args.lr
    pre_epoch = args.pre_epoch
    input_dim = args.input_dim
    n_z = args.n_z
    n_cluster = args.n_cluster

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    x_path = args.x_path
    y_path = args.y_path
    graph_x_path = args.graph_x_path
    save_path = args.save_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int32)
    graph_x = np.loadtxt(graph_x_path, dtype=np.float32)

    encoder = DAEGCEncoder(4, 1, 500, input_dim, n_z, n_cluster,
                       att_dropout=0.0, fdd_dropout=0.0)

    x = torch.from_numpy(feature)
    graph_x = torch.from_numpy(graph_x)

    edge_index_direction = torch.from_numpy(construct_edge_index_direction(graph_x)).t().contiguous()
    edge_index = to_undirected(edge_index_direction, num_nodes=graph_x.shape[0])

    model = GAE(encoder).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for i in range(pre_epoch):
        z, q = model.encode(x, edge_index)
        out_edge = model.decode(z, edge_index)
        loss = F.mse_loss(out_edge, torch.ones_like(out_edge, dtype=torch.float32))
        print("loss:{}".format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), save_path)



