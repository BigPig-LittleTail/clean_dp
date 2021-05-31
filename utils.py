import argparse

import torch
import numpy as np
from torch_geometric.nn import GAE

from model import IDEC, AIDEC, SDCN, TSDCN, DAEGCEncoder, ADAEGCEncoder

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from sklearn.preprocessing import normalize

def common_parameter():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_type", type=str, default="idec")
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--x_path", type=str, default="")
    parser.add_argument("--y_path", type=str, default="")
    parser.add_argument("--pre_path", type=str, default="")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--train_epoch", type=int, default=200)
    parser.add_argument("--input_dim", type=int, default=2000)
    parser.add_argument("--n_z", type=int, default=10)
    parser.add_argument("--n_cluster", type=int, default=30)

    parser.add_argument("--alpha", type=float, default=0.1)

    args, unknown = parser.parse_known_args()

    model_type = args.model_type

    input_dim = args.input_dim
    n_z = args.n_z
    n_cluster = args.n_cluster

    x_path = args.x_path
    y_path = args.y_path
    pre_path = args.pre_path

    seed = args.seed
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    sp_model = False
    if model_type == 'idec':
        TYPE = IDEC
    elif model_type == "aidec":
        TYPE = AIDEC
    elif model_type == "sdcn":
        TYPE = SDCN
    elif model_type == "tsdcn":
        TYPE = TSDCN
    else:
        raise AttributeError("no model")

    feature = np.loadtxt(x_path, dtype=np.float32)
    label = np.loadtxt(y_path, dtype=np.int32)

    model = TYPE(n_enc_1=500,
                 n_enc_2=500,
                 n_enc_3=2000,
                 n_dec_1=2000,
                 n_dec_2=500,
                 n_dec_3=500,
                 n_input=input_dim,
                 n_z=n_z,
                 n_clusters=n_cluster,
                 v=1.0).to(device)

    model.load_pretrain(pre_path)

    x = torch.tensor(feature).to(device)

    z = model.get_ae_hidden(x)

    kmeans = KMeans(n_clusters=n_cluster, n_init=20, random_state=seed)
    predict = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    eva(label, predict, "pretrain")

    return parser, args, model, x, label


def target_distribution(q):
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p


def correct_cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    d = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((d, d), dtype=np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] -= 1
    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost)
    # print(cost[row, col].sum())

    ind = {row[i]: col[i] for i in range(len(row))}
    new_pred = [ind[old] for old in y_pred]

    acc = metrics.accuracy_score(y_true, new_pred)
    f1_macro = metrics.f1_score(y_true, new_pred, average='macro')

    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = correct_cluster_acc(y_true, y_pred)
    # acc_, f1_ = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    # print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
    #         ', f1 {:.4f}'.format(f1))
    print(epoch, ':{:.4f}\t'.format(acc), '{:.4f}\t'.format(nmi), '{:.4f}\t'.format(ari),
            '{:.4f}'.format(f1))


def normalize_adj(A:torch.Tensor):
    A = A + torch.eye(A.size(0))
    d = A.sum(1)
    D = torch.diag(torch.pow(d, -0.5))
    return D.mm(A).mm(D)


def construct_graph(features, method='ncos', topk=10):
    dist = None
    if method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    adj = np.zeros_like(dist, dtype=np.float32)
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        for index in ind:
            if index == i:
                continue
            adj[index][i] = adj[i][index] = 1
    return adj

def construct_edge_index_direction(features, method='ncos', topk=10):
    dist = None
    if method == 'ncos':
        # features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    edge = np.argpartition(dist, -(topk + 1), axis=1)[..., -(topk + 1):, np.newaxis]
    edge = np.insert(edge, 0, np.arange(0, dist.shape[0])[..., np.newaxis], axis=2)
    edge = edge.reshape((-1, 2))
    # 现在是有向图，要转换为无向图
    return edge