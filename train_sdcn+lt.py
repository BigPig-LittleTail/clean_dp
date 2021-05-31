import torch
import torch.nn.functional as F

from torch.optim import Adam

from utils import common_parameter, target_distribution, eva, construct_graph, normalize_adj

if __name__ == '__main__':
    parser, args, model, x, y = common_parameter()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.01)
    args = parser.parse_args(namespace=args)

    lr = args.lr
    train_epoch = args.train_epoch
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    adj = torch.from_numpy(construct_graph(x.data.cpu().numpy()))
    adj = normalize_adj(adj).to(x.device)

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(train_epoch):
        x_bar, q, pred, _, c = model(x, adj)

        if epoch % 1 == 0:
            p = target_distribution(q.data)
            p_c = target_distribution(torch.exp(c).data)

            res1 = q.data.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P
            res3 = torch.exp(c).data.cpu().numpy().argmax(1) # C
            res4 = torch.exp(pred).data.cpu().numpy().argmax(1) # Z

            eva(y, res1, str(epoch) + 'Q')
            eva(y, res4, str(epoch) + 'Z')
            eva(y, res3, str(epoch) + "C")
            eva(y, res2, str(epoch) + 'P')

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        c_loss = F.kl_div(c, p_c, reduction='batchmean')
        ce_loss = F.kl_div(pred, p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, x)

        print("kl_loss{}, re_loss{}, ce_loss{}, c_loss{}".format(kl_loss, re_loss, ce_loss, c_loss))
        loss = re_loss + alpha * kl_loss + beta * c_loss + gamma * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
