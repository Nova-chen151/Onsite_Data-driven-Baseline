import os.path
from typing import Dict
import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from smart.utils.nan_checker import check_nan_inf
from smart.layers.attention_layer import AttentionLayer
from smart.layers import MLPLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from smart.utils import angle_between_2d_vectors
from smart.utils import merge_edges
from smart.utils import weight_init
from smart.utils import wrap_angle
import pickle


class SMARTMapDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 pl2pl_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 map_token) -> None:
        super(SMARTMapDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.pl2pl_radius = pl2pl_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        if input_dim == 2:
            input_dim_r_pt2pt = 3
        elif input_dim == 3:
            input_dim_r_pt2pt = 4
        else:
            raise ValueError('{} is not a valid dimension'.format(input_dim))

        self.type_pt_emb = nn.Embedding(17, hidden_dim)
        self.side_pt_emb = nn.Embedding(4, hidden_dim)
        self.polygon_type_emb = nn.Embedding(4, hidden_dim)
        self.light_pl_emb = nn.Embedding(4, hidden_dim)

        self.r_pt2pt_emb = FourierEmbedding(input_dim=input_dim_r_pt2pt, hidden_dim=hidden_dim,
                                            num_freq_bands=num_freq_bands)
        self.pt2pt_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.token_size = 1024
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        input_dim_token = 22 # 11 * 2
        self.token_emb = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.map_token = map_token
        self.apply(weight_init)
        self.mask_pt = False

    def maybe_autocast(self, dtype=torch.float32):
        return torch.cuda.amp.autocast(dtype=dtype)

    def forward(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pt_valid_mask = data['pt_token']['pt_valid_mask']
        pt_pred_mask = data['pt_token']['pt_pred_mask']
        pt_target_mask = data['pt_token']['pt_target_mask']
        mask_s = pt_valid_mask

        pos_pt = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pt = data['pt_token']['orientation'].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        token_sample_pt = self.map_token['traj_src'].to(pos_pt.device).to(torch.float)
        # token_emb 对地图轨迹进行embedding (pt_token_emb_src: [1024,11,2]->[1024,22]->[1024,128])
        pt_token_emb_src = self.token_emb(token_sample_pt.view(token_sample_pt.shape[0], -1))
        # 根据切分后的多段线token_idx获取token_embedding信息 (pt_token_emb: [n_polyline, 128])
        pt_token_emb = pt_token_emb_src[data['pt_token']['token_idx']]

        if self.input_dim == 2:
            x_pt = pt_token_emb
        elif self.input_dim == 3:
            x_pt = pt_token_emb
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))

        token2pl = data[('pt_token', 'to', 'map_polygon')]['edge_index']
        # 提取每个pt_token对应的信号灯状态
        token_light_type = data['map_polygon']['light_type'][token2pl[1]]
        # type_pt_emb 将token类型进行embedding ([n_polyline]->[n_polyline, 128])
        # polygon_type_emb 将token对应多段线类型进行embedding ([n_polyline]->[n_polyline, 128])
        # light_pl_emb 将token所在多段线灯色类型进行embedding ([n_polyline]->[n_polyline, 128])
        x_pt_categorical_embs = [self.type_pt_emb(data['pt_token']['type'].long()),
                                 self.polygon_type_emb(data['pt_token']['pl_type'].long()),
                                 self.light_pl_emb(token_light_type.long()),]
        # 将地图token_emb, type_pt_emb, polygon_type_emb, light_pl_emb四个结果进行合并（通过sum的方式）
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)
        
        """radius_graph 
        功能：
            函数的主要功能是基于给定的节点特征 x 和半径 r，生成一个图的边索引（edge_index），其中每条边表示一个节点对（两个节点之间的连通性）
        
        传入参数：
            - x([n_polyline, 2]): token节点的坐标和特征
            - r(float): 半径距离，决定两个节点是否可以连接，如果两个节点之间的距离小于 r，则会在它们之间添加一条边【目前为10】
            - batch([n_polyline]): 每个值代表对应节点所属的批次索引，对于每个批次，函数会单独计算邻接关系
            - loop(bool): 指定是否允许自环，即节点是否可以连接到自身
            - max_num_neighbors(int): 限制每个节点的最大邻居数量。仅连接距离最近的 max_num_neighbors 个邻居，避免密集的图结构。
        
        传出参数：
            - edge_index([2, E]) 表示图的边索引，其中 E 是边的数量。edge_index[0, :] 和 edge_index[1, :] 分别表示源节点和目标节点的索引。
        """
        edge_index_pt2pt = radius_graph(x=pos_pt[:, :2], r=self.pl2pl_radius,
                                        batch=data['pt_token']['batch'] if isinstance(data, Batch) else None,
                                        loop=False, max_num_neighbors=100)
        # 根据pt_valid_mask中的有效token重新构建子图
        if self.mask_pt:
            # subgraph 的主要作用是从原始图中选择特定节点，并生成仅包含这些节点的子图。函数会过滤 edge_index 以仅保留连接到这些节点的边。该函数还可以重新编号节点，以确保子图的节点索引是连续的。
            edge_index_pt2pt = subgraph(subset=mask_s, edge_index=edge_index_pt2pt)[0]
        
        # 计算pt2pt各边的相对位置 (rel_pos_pt2pt: [E, 2])
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        # 计算pt2pt各边的朝向夹角（弧度制） (rel_orient_pt2pt: [E])
        rel_orient_pt2pt = wrap_angle(orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]])
        if self.input_dim == 2:
            # torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1) 求出两个token起始点之间的欧几里得距离
            # angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]], nbr_vector=rel_pos_pt2pt[:, :2]) 求出目标token的初始位置向量和相对位置向量之间的夹角（弧度制）
            # r_pt2pt([E, 3]) 表示不同token对之间的相对关系信息
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_orient_pt2pt], dim=-1)
        elif self.input_dim == 3:
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_pos_pt2pt[:, -1],
                 rel_orient_pt2pt], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        # r_pt2pt_emb: 对相对关系信息进行傅里叶嵌入
        # 使用傅里叶嵌入（Fourier Embedding）通常是为了增强模型对复杂非线性模式和周期性特征的表达能力。傅里叶嵌入在特征空间中引入了一系列不同频率的正弦和余弦波，允许模型在特征维度上捕捉更复杂的周期性和振荡模式。
        # 傅里叶嵌入在坐标空间中引入了不同频带（可学习），使得模型可以较好地捕捉空间和时间上的位置偏移。
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        
        # 进行注意力机制计算
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        # 输出头，输出每个被预测位置对应1024个map_token的概率 ([n_preds, 128]->[n_preds, 1024])
        next_token_prob = self.token_predict_head(x_pt[pt_pred_mask])
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        # 每个被预测位置概率最高的10地图token的索引
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)
        next_token_index_gt = data['pt_token']['token_idx'][pt_target_mask]

        return {
            # [n_polyline, 128] 地图token embedding信息（将地图token_emb, type_pt_emb, polygon_type_emb, light_pl_emb通过sum的方式进行合并的结果）
            'x_pt': x_pt,
            # [n_preds, 10] 每个被预测位置概率最高的10地图token的索引
            'map_next_token_idx': next_token_idx,
            # [n_preds, 1024] 每个被预测位置对应1024个map_token的概率
            'map_next_token_prob': next_token_prob,
            # [n_preds] 每个被预测位置对应的真实地图token索引
            'map_next_token_idx_gt': next_token_index_gt,
            # [n_preds] 每个被预测位置的评价掩码，此处全部设置为true
            'map_next_token_eval_mask': pt_pred_mask[pt_pred_mask]
        }
    
    def inference(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        pos_pt = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pt = data['pt_token']['orientation'].contiguous()
        orient_vector_pt = torch.stack([orient_pt.cos(), orient_pt.sin()], dim=-1)
        token_sample_pt = self.map_token['traj_src'].to(pos_pt.device).to(torch.float)
        # token_emb 对地图轨迹进行embedding (pt_token_emb_src: [1024,11,2]->[1024,22]->[1024,128])
        pt_token_emb_src = self.token_emb(token_sample_pt.view(token_sample_pt.shape[0], -1))
        # 根据切分后的多段线token_idx获取token_embedding信息 (pt_token_emb: [n_polyline, 128])
        pt_token_emb = pt_token_emb_src[data['pt_token']['token_idx']]

        if self.input_dim == 2:
            x_pt = pt_token_emb
        elif self.input_dim == 3:
            x_pt = pt_token_emb
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))

        token2pl = data[('pt_token', 'to', 'map_polygon')]['edge_index']
        # 提取每个pt_token对应的信号灯状态
        token_light_type = data['map_polygon']['light_type'][token2pl[1]]
        # type_pt_emb 将token类型进行embedding ([n_polyline]->[n_polyline, 128])
        # polygon_type_emb 将token对应多段线类型进行embedding ([n_polyline]->[n_polyline, 128])
        # light_pl_emb 将token所在多段线灯色类型进行embedding ([n_polyline]->[n_polyline, 128])
        x_pt_categorical_embs = [self.type_pt_emb(data['pt_token']['type'].long()),
                                 self.polygon_type_emb(data['pt_token']['pl_type'].long()),
                                 self.light_pl_emb(token_light_type.long()),]
        # 将地图token_emb, type_pt_emb, polygon_type_emb, light_pl_emb四个结果进行合并（通过sum的方式）
        x_pt = x_pt + torch.stack(x_pt_categorical_embs).sum(dim=0)
        
        edge_index_pt2pt = radius_graph(x=pos_pt[:, :2], r=self.pl2pl_radius,
                                        batch=data['pt_token']['batch'] if isinstance(data, Batch) else None,
                                        loop=False, max_num_neighbors=100)
        
        # 计算pt2pt各边的相对位置 (rel_pos_pt2pt: [E, 2])
        rel_pos_pt2pt = pos_pt[edge_index_pt2pt[0]] - pos_pt[edge_index_pt2pt[1]]
        # 计算pt2pt各边的朝向夹角（弧度制） (rel_orient_pt2pt: [E])
        rel_orient_pt2pt = wrap_angle(orient_pt[edge_index_pt2pt[0]] - orient_pt[edge_index_pt2pt[1]])
        if self.input_dim == 2:
            # torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1) 求出两个token起始点之间的欧几里得距离
            # angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]], nbr_vector=rel_pos_pt2pt[:, :2]) 求出目标token的初始位置向量和相对位置向量之间的夹角（弧度制）
            # r_pt2pt([E, 3]) 表示不同token对之间的相对关系信息
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_orient_pt2pt], dim=-1)
        elif self.input_dim == 3:
            r_pt2pt = torch.stack(
                [torch.norm(rel_pos_pt2pt[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=orient_vector_pt[edge_index_pt2pt[1]],
                                          nbr_vector=rel_pos_pt2pt[:, :2]),
                 rel_pos_pt2pt[:, -1],
                 rel_orient_pt2pt], dim=-1)
        else:
            raise ValueError('{} is not a valid dimension'.format(self.input_dim))
        # r_pt2pt_emb: 对相对关系信息进行傅里叶嵌入
        # 使用傅里叶嵌入（Fourier Embedding）通常是为了增强模型对复杂非线性模式和周期性特征的表达能力。傅里叶嵌入在特征空间中引入了一系列不同频率的正弦和余弦波，允许模型在特征维度上捕捉更复杂的周期性和振荡模式。
        # 傅里叶嵌入在坐标空间中引入了不同频带（可学习），使得模型可以较好地捕捉空间和时间上的位置偏移。
        r_pt2pt = self.r_pt2pt_emb(continuous_inputs=r_pt2pt, categorical_embs=None)
        
        # 进行注意力机制计算
        for i in range(self.num_layers):
            x_pt = self.pt2pt_layers[i](x_pt, r_pt2pt, edge_index_pt2pt)

        return x_pt