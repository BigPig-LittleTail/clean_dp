import torch
import torch.nn.functional as F
import sys
import os
import numpy as  np
from torch.optim import Adam


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model.utils import common_parameter, target_distribution, eva, construct_graph, normalize_adj

if __name__ == '__main__':
    parser, args, model, x, y = common_parameter()
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--graph_x_path", type=str, default="")
    args = parser.parse_args(namespace=args)

    lr = args.lr
    train_epoch = args.train_epoch
    alpha = args.alpha
    gamma = args.gamma
    graph_x_path = args.graph_x_path
    graph_x = np.loadtxt(graph_x_path, dtype=np.float32)

    adj = torch.from_numpy(construct_graph(graph_x))
    adj = normalize_adj(adj).to(x.device)

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(train_epoch):
        x_bar, q, pred, _ = model(x, adj)

        if epoch % 1 == 0:
            p = target_distribution(q.data)

            res1 = q.data.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P
            res4 = torch.exp(pred).data.cpu().numpy().argmax(1) # Z

            eva(y, res1, str(epoch) + 'Q')
            eva(y, res4, str(epoch) + 'Z')
            eva(y, res2, str(epoch) + 'P')

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred, p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, x)

        print("kl_loss{}, re_loss{}, ce_loss{}".format(kl_loss, re_loss, ce_loss))
        loss = re_loss + alpha * kl_loss + gamma * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
