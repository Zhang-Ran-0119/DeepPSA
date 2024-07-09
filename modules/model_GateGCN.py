import torch
import torch.nn as nn
from dgl.readout import sum_nodes
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgl import function as fn
import torch.nn.functional as F
import dgl

class Protac_GateGCN(nn.Module):

    def __init__(self, input_feats,edge_feats,output_feats):
        super(Protac_GateGCN,self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(
            GatedGCNConv(input_feats=input_feats,
                         edge_feats=edge_feats,
                         output_feats=128)
        )
        self.layers.append(
            GatedGCNConv(input_feats=128,
                         edge_feats=128,
                         output_feats=512)
        )
        self.layers.append(
            GatedGCNConv(input_feats=512,
                         edge_feats=512,
                         output_feats=2048)
        )
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 2)
        self.readout=MLPNodeReadout(2048,4096,2048,activation=F.leaky_relu_,mode='mean')
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.weight = WeightedSumAndMax(2048)
        self.gradients = None
        self.g_temp = None

    def activations_hook(self, grad):
        # print("Hook activated")
        # if grad is not None:
        #     print("Gradient received:", grad.shape)
        # else:
        #     print("No gradient received")
        self.gradients = grad

    def forward(self,g,show=False):
        h=g.ndata['atomic']
        e=g.edata['atomic']
        for layer in self.layers:
            h,e=layer(g,h,e)
        if show=="1":
            print("hook")
            h.register_hook(self.activations_hook)

        with g.local_scope():
            g.ndata['h'] = h
            if show == "2":
                g_feats, node_weights = self.weight(g, h, True)
                return g_feats, node_weights
            hg = self.readout(g, h)
            if show == "1":
                hg = F.leaky_relu_(self.fc1(hg))
            else:
                hg = F.leaky_relu_(self.batch_norm1(self.fc1(hg)))
            hg = F.leaky_relu_(self.fc2(hg))
            hg = F.leaky_relu_(self.fc3(hg))
            if show == "1":
                hg = F.leaky_relu_(self.fc3(hg))
            else:
                hg = F.leaky_relu_(self.batch_norm2(self.fc3(hg)))
            temp_g = dgl.graph((g.edges()[0], g.edges()[1]), num_nodes=g.number_of_nodes())
            for key in g.ndata:
                temp_g.ndata[key] = g.ndata[key].clone()
        out = self.fc4(hg)

        if show == "1":
            self.g_temp = temp_g
            return out, self.gradients, self.g_temp
        else:
            return out
    def get_grad_cam_weights(self,g,output, target_category):
        if self.gradients is None:
            raise RuntimeError('No gradients found. Please ensure that the forward method is called with show=True.')

        print("Gradients shape:", self.gradients.shape)  # 打印梯度形状

        device = output.device
        one_hot_output = torch.zeros((1, output.size()[-1]), device=device)
        one_hot_output[0][target_category] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        # 在了解了梯度形状后，正确设置维度
        # pooled_gradients = torch.mean(self.gradients, dim=0)
        # return pooled_gradients
        node_weights = torch.mean(self.gradients, dim=0) * g.ndata['h']
        cam = node_weights.sum(dim=1)
        return cam
class WeightAndSum(nn.Module):
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1)
        )

    def forward(self, g, feats):
        with g.local_scope():
            g.ndata['h'] = feats
            atom_weights = self.atom_weighting(g.ndata['h'])
            g.ndata['w'] = torch.nn.Sigmoid()(self.atom_weighting(g.ndata['h']))
            h_g_sum = sum_nodes(g, 'h', 'w')
        return h_g_sum, atom_weights

class WeightedSumAndMax(nn.Module):

    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats, get_node_weight=False):
        h_g_sum = self.weight_and_sum(bg, feats)[0]
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        atom_weight = self.weight_and_sum(bg, feats)[1]
        if get_node_weight:
            return h_g, atom_weight
        else:
            return h_g



class GatedGCNConv(nn.Module):

    def __init__(
        self,
        input_feats,
        edge_feats,
        output_feats,
        dropout=0,
        batch_norm=True,
        residual=True,
        activation=F.relu,
    ):
        super(GatedGCNConv, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = batch_norm
        self.residual = residual

        if input_feats != output_feats or edge_feats != output_feats:
            self.residual = False

        # Linearly transform the node features.
        self.A = nn.Linear(input_feats, output_feats, bias=True)
        self.B = nn.Linear(input_feats, output_feats, bias=True)
        self.D = nn.Linear(input_feats, output_feats, bias=True)
        self.E = nn.Linear(input_feats, output_feats, bias=True)

        # Linearly transform the edge features.
        self.C = nn.Linear(edge_feats, output_feats, bias=True)

        # Batch normalization on the node/edge features.
        self.bn_node = nn.BatchNorm1d(output_feats)
        self.bn_edge = nn.BatchNorm1d(output_feats)

        self.activation = activation

    def forward(self, graph, feat, edge_feat):
        with graph.local_scope():
            # For residual connection
            h_in = feat
            e_in = edge_feat

            graph.ndata["Ah"] = self.A(feat)
            graph.ndata["Bh"] = self.B(feat)
            graph.ndata["Dh"] = self.D(feat)
            graph.ndata["Eh"] = self.E(feat)
            graph.edata["Ce"] = self.C(edge_feat)

            graph.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))

            # Get edge feature
            graph.edata["e"] = graph.edata["DEh"] + graph.edata["Ce"]
            graph.edata["sigma"] = torch.sigmoid(graph.edata["e"])

            graph.update_all(
                fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
            )
            graph.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
            graph.ndata["h"] = graph.ndata["Ah"] + graph.ndata[
                "sum_sigma_h"
            ] / (graph.ndata["sum_sigma"] + 1e-6)

            # Result of graph convolution.
            feat = graph.ndata["h"]
            edge_feat = graph.edata["e"]

            # Batch normalization.
            if self.batch_norm:
                feat = self.bn_node(feat)
                edge_feat = self.bn_edge(edge_feat)

            # Non-linear activation.
            if self.activation:
                feat = self.activation(feat)
                edge_feat = self.activation(edge_feat)

            # Residual connection.
            if self.residual:
                feat = h_in + feat
                edge_feat = e_in + edge_feat

            feat = self.dropout(feat)
            edge_feat = self.dropout(edge_feat)

            return feat, edge_feat