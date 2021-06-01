import torch.nn.functional as F

from torch.optim import Adam

from model.utils import common_parameter, target_distribution, eva

if __name__ == '__main__':
    parser, args, model, x, y = common_parameter()

    lr = args.lr
    train_epoch = args.train_epoch
    alpha = args.alpha

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(train_epoch):
        x_bar, q = model(x)
        if epoch % 1 == 0:
            p = target_distribution(q.data)

            res1 = q.data.cpu().numpy().argmax(1)  # Q
            res2 = p.data.cpu().numpy().argmax(1)  # P

            eva(y, res1, str(epoch) + 'Q')
            eva(y, res2, str(epoch) + 'P')

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')

        re_loss = F.mse_loss(x_bar, x)

        print("kl_loss{}, re_loss{}".format(kl_loss, re_loss))
        loss = re_loss + alpha * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
