import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset

from typing import Union, Tuple, Optional

from torch_geometric.nn import GAE
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import glorot, zeros

class TopicDataSet(Dataset):
    def __init__(self, x, y, x_topic=None):
        self.x = x
        self.y = y
        self.x_topic = x_topic
        if x_topic is None:
            self.x_topic = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
               torch.from_numpy(np.array(self.x_topic[idx])), \
               torch.from_numpy(np.array(self.y[idx])), \
               torch.from_numpy(np.array(idx))


class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_topic=None, model_type="idec"):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

        if model_type == 'idec' or model_type == 'sdcn':
            self.model_type = model_type
        else:
            raise AttributeError("no this type")

        self.n_topic = n_topic
        if n_topic is not None:
            self.classifier_layer = Linear(n_z, n_topic)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        if self.n_topic:
            c = self.classifier_layer(F.relu(z))
            c = F.log_softmax(c, dim=1)
            if self.model_type == 'idec':
                return x_bar, z, c
            else:
                return x_bar, enc_h1, enc_h2, enc_h3, z, c

        if self.model_type == 'idec':
            return x_bar, z
        else:
            return x_bar, enc_h1, enc_h2, enc_h3, z


class IDEC(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 v=1.0):
        super(IDEC, self).__init__()
        self.v = v

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            n_topic=None,
            model_type="idec")

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def load_pretrain(self, path):
        self.ae.load_state_dict(torch.load(path))

    def get_ae_hidden(self, x):
        with torch.no_grad():
            x_bar, z = self.ae(x)
        return z

    def forward(self, x):
        x_bar, z = self.ae(x)
        # cluster
        q = torch.pow((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2.0), 2) / self.v),
                      -(self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q


class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.mm(adj, support)
        if active:
            output = F.relu(output)
        return output


class SDCN(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            model_type="sdcn"
            )

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def load_pretrain(self, path):
        self.ae.load_state_dict(torch.load(path))

    def get_ae_hidden(self, x):
        with torch.no_grad():
            x_bar, tra1, tra2, tra3, z = self.ae(x)
        return z

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2(h+tra1, adj)
        h = self.gnn_3(h+tra2, adj)
        h = self.gnn_4(h+tra3, adj)
        h = self.gnn_5(h+z, adj, active=False)
        predict = F.log_softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


class AIDEC(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 v=1.0):
        super(AIDEC, self).__init__()
        self.v = v

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            n_topic=n_clusters,
            model_type="idec")

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def load_pretrain(self, path):
        self.ae.load_state_dict(torch.load(path))

    def get_ae_hidden(self, x):
        with torch.no_grad():
            x_bar, z, c = self.ae(x)
        return z

    def forward(self, x):
        x_bar, z, c = self.ae(x)
        q = torch.pow((1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2.0), 2) / self.v),
                      -(self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, c


class TSDCN(nn.Module):
    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z, n_clusters,
                 v=1):
        super(TSDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z,
            n_topic=n_clusters,
            model_type="sdcn"
        )

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def load_pretrain(self, path):
        self.ae.load_state_dict(torch.load(path))

    def get_ae_hidden(self, x):
        with torch.no_grad():
            x_bar, tra1, tra2, tra3, z, c = self.ae(x)
        return z

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z, c = self.ae(x)

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2(h+tra1, adj)
        h = self.gnn_3(h+tra2, adj)
        h = self.gnn_4(h+tra3, adj)
        h = self.gnn_5(h+z, adj, active=False)
        predict = F.log_softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, c



class MGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, order_t: int = 2,  **kwargs):
        super(MGATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.order_t = order_t
        self.M = None

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = alpha_r = (x_l * self.att_l).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        num_nodes = x_l.size(0)
        num_nodes = size[1] if size is not None else num_nodes
        num_nodes = x_r.size(0) if x_r is not None else num_nodes

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        if self.M is None:
            exp = torch.arange(1, self.order_t + 1).to(x_l.device)
            _, col = edge_index
            # D = degree(col, num_nodes).view(-1, 1)
            # B = 1.0 / D.repeat(1, self.order_t)
            # self.M = torch.pow(B, exp).sum(dim=1)
            self.M = torch.pow((1.0 / degree(col, num_nodes).view(-1, 1).repeat(1, self.order_t)), exp).sum(dim=1)\
                .view(-1, 1).repeat(1, H)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, M=(self.M, self.M))

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], M_i: Tensor) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = torch.mul(alpha, M_i)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class DAEGCEncoder(torch.nn.Module):
    def __init__(self, n_head_1, n_head_2, n_hidden, n_input, n_z, n_clusters, att_dropout=0.6, fdd_dropout=0.6, v=1.0):
        super(DAEGCEncoder, self).__init__()
        self.conv1 = MGATConv(n_input, n_hidden, heads=n_head_1, dropout=att_dropout)
        self.conv2 = MGATConv(n_hidden * n_head_1, n_z, heads=n_head_2, concat=False, dropout=att_dropout)

        self.fdd_dropout = fdd_dropout

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.fdd_dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.fdd_dropout, training=self.training)
        x = self.conv2(x, edge_index)

        q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x, q


class ADAEGCEncoder(torch.nn.Module):
    def __init__(self, n_head_1, n_head_2, n_hidden, n_input, n_z, n_clusters, att_dropout=0.6, fdd_dropout=0.6, v=1.0, n_topic=None):
        super(ADAEGCEncoder, self).__init__()
        self.conv1 = MGATConv(n_input, n_hidden, heads=n_head_1, dropout=att_dropout)
        self.conv2 = MGATConv(n_hidden * n_head_1, n_z, heads=n_head_2, concat=False, dropout=att_dropout)
        self.fdd_dropout = fdd_dropout

        if not n_topic:
            n_topic = n_clusters

        # self.classifier_layer = GATConv(n_z * n_head_2, n_clusters, heads=1, concat=False, dropout=att_dropout)
        self.classifier_layer = Linear(n_z, n_clusters)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.fdd_dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.fdd_dropout, training=self.training)
        z = self.conv2(x, edge_index)

        # c = self.classifier_layer(F.relu(z), edge_index)
        c = self.classifier_layer(F.relu(z))
        c = F.log_softmax(c, dim=1)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return z, q, c
