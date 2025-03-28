from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims=2):
        if dims == 2:
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
        elif dims == 3:
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
        else:
            print("nconv dimension error")
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop_gated_attention(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, device):
        super(mixprop_gated_attention, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.W_a = nn.Parameter(torch.randn(c_in, c_in).to(device), requires_grad=True).to(device)
        self.W_b = nn.Parameter(torch.randn(c_in, c_in).to(device), requires_grad=True).to(device)
        self.W = nn.Parameter(torch.randn(c_out, c_out).to(device), requires_grad=True).to(device)
        self.v = nn.Parameter(torch.randn(1, c_out).to(device), requires_grad=True).to(device)
        self.u = nn.Parameter(torch.randn(1, c_in).to(device), requires_grad=True).to(device)

    def forward(self, h_in, adj):
        adj = adj.to(h_in.device).float()
        h = h_in
        out = [h]
        n = h_in.shape[1]
        for i in range(self.gdep):  # information propagation
            e_in_1 = self.nconv(h_in.transpose(1, 2), self.W_a, 2)
            e_in = self.nconv(e_in_1,  self.u, 2).transpose(1, 2)
            e_l_1 = self.nconv(h.transpose(1, 2), self.W_b, 2)
            e_l = self.nconv(e_l_1, self.u, 2).transpose(1, 2)
            stack_e = torch.cat((e_l, e_in), dim=1)
            softmax_e = torch.softmax(stack_e, dim=1)
            alpha_k = softmax_e[:, 0:1, :, :]
            h = (1 - alpha_k) * h_in + alpha_k * self.nconv(h, adj, adj.dim())
            h = F.relu(self.mlp(h))
            out.append(h)
        e = torch.cat([self.nconv(self.nconv(h_k.transpose(1, 2), self.W, 2),
                                  self.v, 2) for h_k in out], dim=2).transpose(1, 2)
        beta = torch.softmax(e, dim=1)
        h_out = sum(beta[:, k*n:(k+1)*n, :, :] * out[k] for k in range(self.gdep + 1))
        return h_out


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [3, 5, 7, 9]
        assert cout % len(self.kernel_set) == 0, 'adjust length of kernel set or conv_channels'
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            padding = ((kern - 1) * dilation_factor) // 2
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor), padding=(0, padding)))

    def forward(self, input):
        x = [tconv(input) for tconv in self.tconv]
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, pred_definedA, alpha=3, mask_bool=True):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.mask = pred_definedA
        self.mask_bool = mask_bool

    def forward(self, idx):
        nodevec2 = self.emb2(idx)
        nodevec1 = self.emb1(idx)
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        if self.mask_bool:
            adj = F.softmax(torch.tanh(self.alpha * a), dim=1)
            adj = adj * self.mask
        else:
            adj = F.softmax(F.relu(self.alpha * a) + self.mask, dim=1)
        return adj

    def fullA(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.softmax(torch.tanh(self.alpha * a), dim=1)
        return adj


class adaptive_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, pred_definedA, alpha=3, mask_bool=True):
        super(adaptive_graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.emb1 = nn.Embedding(nnodes, dim)
        self.emb2 = nn.Embedding(nnodes, dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.mask = pred_definedA
        self.mask_bool = mask_bool

    def forward(self, idx):
        nodevec2 = self.emb2(idx)
        nodevec1 = self.emb1(idx)
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        if self.mask_bool:
            adj = F.softmax(torch.tanh(self.alpha * a), dim=1)
            mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
            mask.fill_(float('0'))
            s1, t1 = adj.topk(self.k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj*mask
        else:
            adj = F.softmax(F.relu(self.alpha * a), dim=1)
        return adj

    def fullA(self, idx):
        nodevec1 = self.emb1(idx)
        nodevec2 = self.emb2(idx)

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.softmax(torch.tanh(self.alpha * a), dim=1)
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size1=10, hidden_size2=20, hidden_size3=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class gtnet_Signal(nn.Module):
    def __init__(self, seq_in, gcn_depth, num_nodes, device,
                 cycle_num, predefined_A=None, dropout=0.3,
                 subgraph_size=20, node_dim=40, dilation_exponential=2,
                 conv_channels=32, residual_channels=32, skip_channels=32,
                 end_channels=128, seq_length=12, in_dim=1, out_dim=12,
                 layers=2, tanhalpha=3, layer_norm_affline=True, mlp_indim=31,
                 adap_only=False, dynamic_bool=True, gamma=0.5):
        super(gtnet_Signal, self).__init__()
        self.cycle_num = cycle_num
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.filter_convs2 = nn.ModuleList()
        self.gate_convs2 = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.ModuleList()
        self.dynamic_bool = dynamic_bool
        self.add_skip = True
        mask_bool = True
        self.gamma = gamma
        if self.dynamic_bool:
            self.adpvec = nn.Parameter(torch.randn(seq_in, seq_in).to(device), requires_grad=True).to(device)
        for i in range(3):
            self.start_conv.append(nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1)))
        if adap_only:
            self.gc = adaptive_graph_constructor(num_nodes, subgraph_size, node_dim, device, self.predefined_A, alpha=tanhalpha, mask_bool=mask_bool)
        else:
            self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, self.predefined_A, alpha=tanhalpha, mask_bool=mask_bool)
        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.filter_convs2.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs2.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, 1)))
                self.gconv1.append(mixprop_gated_attention(conv_channels, residual_channels, gcn_depth, dropout, device))
                self.gconv2.append(mixprop_gated_attention(conv_channels, residual_channels, gcn_depth, dropout, device))
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length+self.cycle_num+1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential
        self.skipE = nn.Conv2d(in_channels=conv_channels, out_channels=skip_channels, kernel_size=(1, 1))
        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp1 = MLP(input_size=mlp_indim, output_size=mlp_indim)
        self.mlp2 = MLP(input_size=mlp_indim, output_size=1)

    def forward(self, input, idx=None):
        input = input.transpose(2, 3)
        seq_len = input.size(3)
        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length,0,0,0))

        # adaptive graph learning layer
        if idx is None:
            adp = self.gc(self.idx)
        else:
            adp = self.gc(idx)

        if self.dynamic_bool:
            xn = input[:, 0, :, :]
            xn = (xn - xn.min(dim=-1)[0].unsqueeze(-1)) / \
                 (xn.max(dim=-1)[0] - xn.min(dim=-1)[0]).unsqueeze(-1)
            xn = torch.nan_to_num(xn, nan=0.5)
            xn = xn / torch.sqrt((xn ** 2).sum(dim=-1)).unsqueeze(-1)
            adp_dynamic = torch.einsum('nvt, tc->nvc', (xn, self.adpvec))
            adp_dynamic = torch.bmm(adp_dynamic, xn.permute(0, 2, 1))
            adp_dynamic = F.softmax(F.relu(adp_dynamic), dim=1)
            adp = self.gamma * adp_dynamic + (1 - self.gamma) * adp  # dynamic + static

        skip = 0
        x = input
        x1 = input[:, 0:1, :, :]
        x2 = input[:, 1:2, :, :self.cycle_num]
        x3 = input[:, 2:3, :, 0:1]
        x1 = self.start_conv[0](x1)
        x2 = self.start_conv[1](x2)
        x3 = self.start_conv[2](x3)
        x = torch.cat([x1, x2, x3], dim=-1)
        for i in range(self.layers):
            residual = x
            x1 = x[..., :seq_len]
            x2 = x[..., seq_len:seq_len+self.cycle_num]
            x3 = x[..., seq_len+self.cycle_num:]
            # TC Module
            filter1 = self.filter_convs[i](x1)
            filter1 = torch.tanh(filter1)
            gate1 = self.gate_convs[i](x1)
            gate1 = torch.sigmoid(gate1)
            x1 = filter1 * gate1

            filter2 = self.filter_convs2[i](x2)
            filter2 = torch.tanh(filter2)
            gate2 = self.gate_convs2[i](x2)
            gate2 = torch.sigmoid(gate2)
            x2 = filter2 * gate2
            x = torch.cat([x1, x2, x3], dim=-1)
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip += s

            # GC Module
            x1 = x[..., :seq_len]
            if self.dynamic_bool:
                if self.add_skip:
                    x1 = (self.gconv1[i](x1, adp) + self.gconv2[i](x1, adp.transpose(1,2))) + x1
                else:
                    x1 = (self.gconv1[i](x1, adp) + self.gconv2[i](x1, adp.transpose(1,2)))
            else:
                if self.add_skip:
                    x1 = (self.gconv1[i](x1, adp) + self.gconv2[i](x1, adp.transpose(0,1))) + x1
                else:
                    x1 = (self.gconv1[i](x1, adp) + self.gconv2[i](x1, adp.transpose(0,1)))
            x = torch.cat([x1, x[..., seq_len:]], dim=3)
            x = x + residual
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
        skip += self.skipE(x)
        skip = self.mlp2(skip)
        x = F.relu(self.end_conv_1(skip))
        x = self.end_conv_2(x)
        x = x.transpose(2, 3)
        return x
