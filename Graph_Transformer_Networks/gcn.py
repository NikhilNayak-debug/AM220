import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.utils import add_self_loops
from inits import glorot, zeros
import dgl
from scipy import sparse as sp
import numpy as np
import torch.nn.functional as F

class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 args=None):
        super(GCNConv, self).__init__('add', flow='target_to_source')

        self.in_channels = in_channels + 16
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.args = args
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    def laplace_decomp(self, g, max_freqs=10):

        # Laplacian
        n = g.number_of_nodes()
        A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(g.number_of_nodes()) - N * A * N

        # Eigenvectors with numpy
        EigVals, EigVecs = np.linalg.eigh(L.toarray())
        EigVals, EigVecs = EigVals[: max_freqs], EigVecs[:,
                                                 :max_freqs]  # Keep up to the maximum desired number of frequencies

        # Normalize and pad EigenVectors
        EigVecs = torch.from_numpy(EigVecs).float()
        EigVecs = F.normalize(EigVecs, p=2, dim=1, eps=1e-12, out=None)

        if n < max_freqs:
            g.ndata['EigVecs'] = F.pad(EigVecs, (0, max_freqs - n), value=float('nan'))
        else:
            g.ndata['EigVecs'] = EigVecs

        # Save eigenvales and pad
        EigVals = torch.from_numpy(np.sort(np.abs(np.real(
            EigVals))))  # Abs value is taken because numpy sometimes computes the first eigenvalue approaching 0 from the negative

        if n < max_freqs:
            EigVals = F.pad(EigVals, (0, max_freqs - n), value=float('nan')).unsqueeze(0)
        else:
            EigVals = EigVals.unsqueeze(0)

        # Save EigVals node features
        g.ndata['EigVals'] = EigVals.repeat(g.number_of_nodes(), 1).unsqueeze(2)

        return EigVecs, EigVals


    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None, args=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        loop_weight = torch.full((num_nodes, ),
                                1 if not args.remove_self_loops else 0,
                                dtype=edge_weight.dtype,
                                device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        
        # deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # return edge_index, (deg_inv_sqrt[col] ** 0.5) * edge_weight * (deg_inv_sqrt[row] ** 0.5)
        return edge_index, deg_inv_sqrt[row] * edge_weight


    def forward(self, x, edge_index, edge_weight=None, g=None):
        """"""

        EigVecs, EigVals = self.laplace_decomp(g[0])

        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float()  # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc)  # (Num nodes) x (Num Eigenvectors) x 2

        PosEnc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0, 1).float()  # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc)  # (Num Eigenvectors) x (Num nodes) x PE_dim

        # 1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:, :, 0])

        # remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0, 1)[:, :, 0]] = float('nan')

        # Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)

        # Concatenate learned PE to input embedding
        x = torch.cat((x, PosEnc), 1)

        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype, args=self.args)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)