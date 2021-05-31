import argparse

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from model import AE, TopicDataSet
from sklearn.preprocessing import normalize
from sklearn.decomposition import NMF


def nmf(x, n_components, seed=None):
    model = NMF(n_components=n_components, init='random', random_state=seed, max_iter=1000)
    w = model.fit_transform(x).astype(dtype=np.float32)
    return normalize(w, norm="l1", axis=1)


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
    parser.add_argument("--n_topic", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle", type=str, default="True")

    args = parser.parse_args()

    lr = args.lr
    pre_epoch = args.pre_epoch
    input_dim = args.input_dim
    n_z = args.n_z
    n_topic = args.n_topic
    batch_size = args.batch_size

    alpha = args.alpha
    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True


    if args.shuffle == "True":
        shuffle = True
    else:
        shuffle = False

    x_path = args.x_path
    y_path = args.y_path
    save_path = args.save_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature = np.loadtxt(x_path, dtype=np.float32)
    label = np.loadtxt(y_path, dtype=np.int32)

    if n_topic:
        topic = nmf(feature, n_topic, seed=seed)
        dataset = TopicDataSet(feature, label, x_topic=topic)
    else:
        dataset = TopicDataSet(feature, label)

    model = AE(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=2000,
        n_dec_1=2000,
        n_dec_2=500,
        n_dec_3=500,
        n_input=input_dim,
        n_z=n_z,
        n_topic=n_topic,
        model_type="idec"
    ).to(device)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(pre_epoch):

        for batch_idx, (x, x_topic, _, _) in enumerate(train_loader):
            x = x.to(device)

            if n_topic:
                x_topic = x_topic.to(device)
                x_bar, z, c = model(x)
                re_loss = F.mse_loss(x_bar, x)
                kl_loss = F.kl_div(c, x_topic, reduction='batchmean')
                loss = re_loss + alpha * kl_loss
            else:
                x_bar, z = model(x)
                loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch, loss))
        torch.save(model.state_dict(), save_path)
    print("model saved to {}".format(save_path))