import pickle
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math
import time


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class SMARTAgentDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_data: Dict,
                 token_size=512) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim)

        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )   # 自注意力机制
        self.pt2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )   # 交叉注意力机制
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )   # 自注意力机制
        self.token_size = token_size
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        self.trajectory_token = token_data['token']
        self.trajectory_token_traj = token_data['traj']
        self.trajectory_token_all = token_data['token_all']
        self.apply(weight_init)
        self.shift = 5
        self.beam_size = 5
        self.hist_mask = True

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel

    def agent_token_embedding(self, data, agent_category, agent_token_index, pos_a, head_vector_a, inference=False):
        num_agent, num_step, traj_dim = pos_a.shape
        
        # motion_vector_a([NA, 18, 2]) 轨迹位移向量，第一个索引为[0,0]
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)
        cyc_mask = (agent_type == 2)
        ped_mask = (agent_type == 1)

        # 聚类后的轨迹token embedding {MLPEmbedding}
        trajectory_token_veh = torch.from_numpy(self.trajectory_token['veh']).clone().to(pos_a.device).to(torch.float)  # [2048, 4, 2]
        self.agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh.view(trajectory_token_veh.shape[0], -1))     # [2048, 128]
        trajectory_token_ped = torch.from_numpy(self.trajectory_token['ped']).clone().to(pos_a.device).to(torch.float)  # [2048, 4, 2]
        self.agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped.view(trajectory_token_ped.shape[0], -1))     # [2048, 128]
        trajectory_token_cyc = torch.from_numpy(self.trajectory_token['cyc']).clone().to(pos_a.device).to(torch.float)  # [2048, 4, 2]
        self.agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1))     # [2048, 128]

        if inference:
            agent_token_traj_all = torch.zeros((num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device) # [NA, 2048, 6, 4, 2]
            trajectory_token_all_veh = torch.from_numpy(self.trajectory_token_all['veh']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_ped = torch.from_numpy(self.trajectory_token_all['ped']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_cyc = torch.from_numpy(self.trajectory_token_all['cyc']).clone().to(pos_a.device).to(
                torch.float)
            agent_token_traj_all[veh_mask] = torch.cat(
                [trajectory_token_all_veh[:, :self.shift], trajectory_token_veh[:, None, ...]], dim=1)
            agent_token_traj_all[ped_mask] = torch.cat(
                [trajectory_token_all_ped[:, :self.shift], trajectory_token_ped[:, None, ...]], dim=1)
            agent_token_traj_all[cyc_mask] = torch.cat(
                [trajectory_token_all_cyc[:, :self.shift], trajectory_token_cyc[:, None, ...]], dim=1)

        # agent_token_emb([NA, 18, 128]) 将agent_token与上述处理好的traj_token对应
        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        # agent_token_traj([NA, 18, 2048, 4, 2]) agent_token对应的轨迹
        agent_token_traj = torch.zeros((num_agent, num_step, self.token_size, 4, 2), device=pos_a.device)
        agent_token_traj[veh_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyc_mask] = trajectory_token_cyc

        vel = data['agent']['token_velocity']

        categorical_embs = [
            # type_a_emb {nn.Embedding}
            # data['agent']['type'] ([NA]) -> self.type_a_emb() ([NA, 128]) -> .repeat_interleave() ([NA * 18, 128])
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=num_step,
                                                                            dim=0),
            # shape_emb {MLPLayer}
            # data['agent']['shape'][:, self.num_historical_steps - 1, :] ([NA, 3]) -> self.shape_emb() ([NA, 128]) -> .repeat_interleave() ([NA * 18, 128])
            self.shape_emb(data['agent']['shape'][:, self.num_historical_steps - 1, :]).repeat_interleave(
                repeats=num_step,
                dim=0)
        ]
        
        # feature_a ([NA, 18, 2]) agent的行驶特征（包含token间距离和航向角变化）
        feature_a = torch.stack(
            [
                # [NA, 18] 各agent_token到前一个token的距离
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                # [NA, 18] 各agent_token当前朝向(head_vector_a)与到下一位置航向(motion_vector_a)间的夹角
                angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])
            ]
            , dim=-1)

        # x_a_emb {FourierEmbedding}
        # 先对feature_a进行FourierEmbedding，之后通过sum方式融合categorical_embs中的两个属性相关的embedding向量，最后通过一层线性变换进行输出
        x_a = self.x_a_emb(continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
                           categorical_embs=categorical_embs)
        x_a = x_a.view(-1, num_step, self.hidden_dim)   # [NA * 18, 128] -> [NA, 18, 128]

        # fusion_emb {MLPEmbedding} [35, 18, 256] -> [35, 18, 128]
        # feat_a 整合代理的属性和token信息的最后一层嵌入向量
        # - agent_token_emb 对轨迹token进行词嵌入
        # - x_a 融合代理类型、代理形状、代理行驶特征（token间距离和航向角变化）进行词嵌入
        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs
        else:
            return feat_a, agent_token_traj

    def agent_predict_next(self, data, agent_category, feat_a):
        num_agent, num_step, traj_dim = data['agent']['token_pos'].shape
        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        token_res = torch.zeros((num_agent, num_step, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])
        return token_res

    def agent_predict_next_inf(self, data, agent_category, feat_a):
        num_agent, traj_dim = feat_a.shape
        agent_type = data['agent']['type']

        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3

        token_res = torch.zeros((num_agent, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])

        return token_res

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None):
        pos_t = pos_a.reshape(-1, self.input_dim)       # [NA * 18, 2]
        head_t = head_a.reshape(-1)                     # [NA * 18]    
        head_vector_t = head_vector_a.reshape(-1, 2)    # [NA * 18, 2]
        hist_mask = mask.clone()                        # [NA, 18]

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1), torch.randint(0, mask.shape[1], (num_agent, 10))] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            # mask_t [NA, 18, 18] 对每个代理包含了一个TxT的矩阵，其中每个元素 (i, j) 表示原掩码中第 i 时间步和第 j 时间步的实体是否都有效
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        # edge_index_t ([2, E_t]) 同一代理不同有效时刻之间建立边
        # dense_to_sparse 函数的作用是将 mask_t 这种稠密的布尔矩阵转换为稀疏表示形式
        # edge_index中的边索引范围是 0 - 18 * NA 
        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        # 筛选出有效时刻在time_span(30帧)之内的边
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift]
        
        # 同一代理不同有效时刻之间的相对位置
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        # 同一代理不同有效时刻之间的相对朝向
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack([
                # [E_t] 筛选出的同一代理不同时刻间的距离
                torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
                # [E_t] 当前航向与位移向量之间夹角
                angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
                # [E_t] 同一代理不同时刻间的航向角之差
                rel_head_t,
                # [E_t] token所在时刻的索引之差(-1表示相差5帧，-2表示相差10帧 依此类推)
                edge_index_t[0] - edge_index_t[1]
            ], dim=-1)
        # r_t_emb {FourierEmbedding}
        # r_t ([E_t, 128]) 对同一代理在时间窗time_span内的不同有效时刻间信息进行embedding
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)       # [NA * 18, 2] 所有代理每一步的位置
        head_s = head_a.transpose(0, 1).reshape(-1)                     # [NA * 18] 所有代理每一步的航向
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)    # [NA * 18, 2] 所有代理每一步的航向向量
        # 在同一时刻中，位置欧几里得距离小于a2a_radius（60）的代理之间建立边
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        # 根据mask_s删除图中的无效边
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]              # [2, E_a2a] 同一时刻不同代理之间的有效边
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]                   # [E_a2a, 2] 代理之间位置差（1指向0）
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])    # [E_a2a] 代理1航向转向代理0航向的有向角度
        r_a2a = torch.stack([
                # [E_a2a] 同一时刻关联代理之间的距离
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                # [E_a2a] 同一时刻 两关联向量间位移向量与关联代理1航向 之间的夹角
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
                # [E_a2a] 同一时刻关联代理之间的航向差
                rel_head_a2a
            ], dim=-1)
        # r_a2a_emb {FourierEmbedding}
        # r_a2a ([E_a2a, 128]) 对同一时刻不同关联代理之间的信息进行embedding
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(self, data, num_step, pos_a, head_a, head_vector_a, mask,
                             batch_s, batch_pl):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)                       # [NA * 18]
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)               # [NA * 18, 2]
        head_s = head_a.transpose(0, 1).reshape(-1)                             # [NA * 18]
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)            # [NA * 18, 2]
        pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)                                     # [n_polyline * 18, 2]
        orient_pl = orient_pl.repeat(num_step)                                  # [n_polyline * 18]
        # 同一时刻中，在欧几里得距离小于pl2a_radius（30）的代理和多段线之间建立边 [polyline_idx, agent_idx]
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                 batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        # 筛除包含无效代理的一些边
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]                         # [2, E_pl2a] 同一时刻代理与多段线之间的有效边
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]                       # [E_pl2a, 2] 代理指向多段线坐标的位移向量
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])    # [E_pl2a] 代理航向角转到多段线朝向的有向角度
        r_pl2a = torch.stack([
                # [E_pl2a] 同一时刻代理到多段线的距离
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                # [E_pl2a] 同一时刻 代理航向向量 转到 代理指向多段线坐标的位移向量 的角度
                angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
                # [E_pl2a] 代理航向角转到多段线朝向的有向角度
                rel_orient_pl2a
            ], dim=-1)
        # r_pt2a_emb {FourierEmbedding}
        # r_pl2a ([E_pl2a, 128]) 对同一时刻代理与关联多段线间的信息进行embedding
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_a = data['agent']['token_pos']                                  # [NA, 18, 2]
        head_a = data['agent']['token_heading']                             # [NA, 18]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)   # [NA, 18, 2]
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data['agent']['category']                          # [NA]
        agent_token_index = data['agent']['token_idx']                      # [NA, 18]
        
        # feat_a 表示对agent信息进行编码后的向量 [NA, 18, 128]
        # - 融合 轨迹token、代理类型、代理形状、代理行驶特征（token间距离和航向角变化） 进行词嵌入
        # agent_token_traj 表示记录了每个token对应的轨迹信息 [NA, 18, 2048, 4, 2]
        feat_a, agent_token_traj = self.agent_token_embedding(data, agent_category, agent_token_index,
                                                              pos_a, head_vector_a)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        # agent_valid_mask[~eval_mask] = False
        mask = agent_valid_mask
        # edge_index_t ([2, E_t]) 同一代理在时间窗time_span内的不同有效时刻间建立边
        # r_t ([E_t, 128]) 对同一代理在时间窗time_span内的不同有效时刻间信息进行embedding
        # - 融合 同一代理不同时刻间的距离、当前航向与位移向量之间夹角、同一代理不同时刻间的航向角之差、token所在时刻的索引之差 进行词嵌入
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        if isinstance(data, Batch):
            # [NA * 18]
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(num_step)], dim=0)
            # [n_polyline * 18]
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                  for t in range(num_step)], dim=0)
        else:
            batch_s = torch.arange(num_step,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(num_step,
                                    device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])

        mask_s = mask.transpose(0, 1).reshape(-1)     # [NA * 18]
        # edge_index_a2a ([2, E_a2a])  同一时刻位置在60米范围内的关联代理之间建立边
        # r_a2a ([E_a2a, 128]) 对同一时刻不同关联代理之间的信息进行embedding
        # - 融合 同一时刻关联代理之间的距离、航向差、位移向量与关联代理1航向的夹角 进行词嵌入
        edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)
        
        # 筛选出被预测代理
        mask[agent_category != 3] = False
        # edge_index_pl2a ([2, E_pl2a]) 同一时刻代理与多段线之间的有效边
        # r_pl2a ([E_pl2a, 128]) 对同一时刻代理与关联多段线间的信息进行embedding
        # - 融合 同一时刻代理到多段线的距离、代理航向向量 转到 代理指向多段线坐标的位移向量 的角度、代理航向角转到多段线朝向的有向角度 进行词嵌入
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, pos_a, head_a,
                                                            head_vector_a, mask, batch_s, batch_pl)

        for i in range(self.num_layers):
            # 同一代理不同时刻之间进行自注意力机制
            feat_a = feat_a.reshape(-1, self.hidden_dim)                # [NA, 18, 12] -> [18 * NA, 128]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)   # [18 * NA, 128]
            
            # 同一时刻代理与关联多段线之间进行交叉注意力机制
            feat_a = feat_a.reshape(-1, num_step,
                                    self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)   # [18 * NA, 128] -> [NA * 18, 128]
            feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                    -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            
            # 同一时刻不同关联代理之间进行自注意力机制
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)         # [NA * 18, 128]
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)  # [NA, 18, 128]

        num_agent, num_step, hidden_dim, traj_num, traj_dim = agent_token_traj.shape
        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        next_token_eval_mask[:, -1] = False

        return {
            # [NA, 18, 128] 喂入token_predict_head进行预测前的agent特征向量
            'x_a': feat_a,
            # [NA, 18, 10] 每一个代理每一个预测步预测得到的top10的token编号
            'next_token_idx': next_token_idx,
            # [NA, 18, 2048] 每一个代理每一个预测步预测得到的2048个token的概率
            'next_token_prob': next_token_prob,
            # [NA, 18] 每一个代理每一个预测步真实匹配到的token编号
            'next_token_idx_gt': next_token_index_gt,
            # [NA, 18] 每一个代理每一个预测步是否进行评价
            'next_token_eval_mask': next_token_eval_mask,
        }

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor],
                  show_detail: bool=False) -> Dict[str, torch.Tensor]:
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]       # [NA] 提取第11帧的代理有效情况
        pos_a = data['agent']['token_pos'].clone()                                      # [NA, 18, 2] token对应的矩形中心点坐标
        head_a = data['agent']['token_heading'].clone()                                 # [NA, 18] token对应航向角
        num_agent, num_step, traj_dim = pos_a.shape
        pos_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        head_a[:, (self.num_historical_steps - 1) // self.shift:] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)               # [NA, 18, 2] token对应航向向量

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()                    # [NA, 18] 每个token时间步是否有效
        agent_valid_mask[:, (self.num_historical_steps - 1) // self.shift:] = True
        agent_valid_mask[~eval_mask] = False
        agent_token_index = data['agent']['token_idx']                                  # [NA, 18] 每个时间步对应的token索引
        agent_category = data['agent']['category']                                      # [NA] 每个代理的类型（3为被预测代理） 
        feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs = self.agent_token_embedding(
            data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True)

        agent_type = data["agent"]["type"]
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        av_mask = data["agent"]["av_index"]

        self.num_recurrent_steps_val = data["agent"]['position'].shape[1]-self.num_historical_steps                         # 80
        pred_traj = torch.zeros(data["agent"]['num_nodes'], self.num_recurrent_steps_val, 2, device=feat_a.device)             # [NA, 80, 2], 各代理每一帧的预测中心点坐标
        pred_head = torch.zeros(data["agent"]['num_nodes'], self.num_recurrent_steps_val, device=feat_a.device)                # [NA, 80], 各代理每一帧的预测航向角
        pred_prob = torch.zeros(data["agent"]['num_nodes'], self.num_recurrent_steps_val // self.shift, device=feat_a.device)  # [NA, 16], 各代理在各时刻所采样token的分布概率
        next_token_idx_list = []            # 记录每个预测步各代理预测的token索引
        mask = agent_valid_mask.clone()
        inference_time = []
        feat_a_t_dict = {}
        for t in range(self.num_recurrent_steps_val // self.shift):
            if t == 0:
                inference_mask = mask.clone()
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t:] = False
            else:
                inference_mask = torch.zeros_like(mask)
                inference_mask[:, (self.num_historical_steps - 1) // self.shift + t - 1] = True
            tic = time.time()
            edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask, inference_mask)
            if isinstance(data, Batch):
                batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                     for t in range(num_step)], dim=0)
                batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                      for t in range(num_step)], dim=0)
            else:
                batch_s = torch.arange(num_step,
                                       device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
                batch_pl = torch.arange(num_step,
                                        device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])
            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, pos_a, head_a,
                                                                head_vector_a,
                                                                inference_mask, batch_s,
                                                                batch_pl)
            mask_s = inference_mask.transpose(0, 1).reshape(-1)
            edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a,
                                                                batch_s, mask_s)

            for i in range(self.num_layers):
                if i in feat_a_t_dict:
                    feat_a = feat_a_t_dict[i]
                feat_a = feat_a.reshape(-1, self.hidden_dim)                                            # [NA, 18, 128] -> [18 * NA, 128]
                feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
                feat_a = feat_a.reshape(-1, num_step,
                                        self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)   # [18 * NA, 128] -> [NA * 18, 128]
                feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                    repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                        -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
                feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
                feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)                  # [NA * 18, 128] -> [NA, 18, 128]

                if i+1 not in feat_a_t_dict:
                    feat_a_t_dict[i+1] = feat_a
                else:
                    feat_a_t_dict[i+1][:, (self.num_historical_steps - 1) // self.shift - 1 + t] = feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]

            next_token_prob = self.token_predict_head(feat_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]) # [NA, 2048]

            next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)                                            # [NA, 2048]

            topk_prob, next_token_idx = torch.topk(next_token_prob_softmax, k=self.beam_size, dim=-1)                   # [NA, 5(beam_size)]

            expanded_index = next_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2)
            next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index)                                     # [NA, 5, 6, 4, 2]
            
            toc = time.time()
            inference_time.append((toc - tic)*1000/5)

            theta = head_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            # 将token轨迹转换到全局坐标系
            agent_diff_rel = torch.bmm(next_token_traj.view(-1, 4, 2),
                                       rot_mat[:, None, None, ...].repeat(1, self.beam_size, self.shift + 1, 1, 1).view(
                                           -1, 2, 2)).view(num_agent, self.beam_size, self.shift + 1, 4, 2)
            agent_pred_rel = agent_diff_rel + pos_a[:, (self.num_historical_steps - 1) // self.shift - 1 + t, :][:, None, None, None, ...]  # [NA, 5, 6, 4, 2]

            sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device)                                    # [NA, 1] 根据topk_prob进行采样
            agent_pred_rel = agent_pred_rel.gather(dim=1,
                                                   index=sample_index[..., None, None, None].expand(-1, -1, 6, 4,
                                                                                                    2))[:, 0, ...]      # [NA, 6, 4, 2] 根据采样结果得到token的全局轨迹
            pred_prob[:, t] = topk_prob.gather(dim=-1, index=sample_index)[:, 0]                    # [NA] 更新t时刻个代理选取token的概率
            pred_traj[:, t * 5:(t + 1) * 5] = agent_pred_rel[:, 1:, ...].clone().mean(dim=2)        # [NA, 5, 2] 更新未来5帧的预测轨迹中心点坐标
            diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
            pred_head[:, t * 5:(t + 1) * 5] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])     # [NA, 5] 更新未来5帧的预测航向角

            pos_a[:, (self.num_historical_steps - 1) // self.shift + t] = agent_pred_rel[:, -1, ...].clone().mean(dim=1)    # [NA, 2] 更新第t步的车辆位置
            diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])                     # [NA] 更新第t步的车辆航向
            head_a[:, (self.num_historical_steps - 1) // self.shift + t] = theta
            next_token_idx = next_token_idx.gather(dim=1, index=sample_index)
            next_token_idx = next_token_idx.squeeze(-1)
            next_token_idx_list.append(next_token_idx[:, None])                     # [NA, 1] 添加token索引
            
            # 更新第t步的agent_token_emb
            agent_token_emb[veh_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_veh[
                next_token_idx[veh_mask]]
            agent_token_emb[ped_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_ped[
                next_token_idx[ped_mask]]
            agent_token_emb[cyc_mask, (self.num_historical_steps - 1) // self.shift + t] = self.agent_token_emb_cyc[
                next_token_idx[cyc_mask]]
            
            # 更新feat_a (与agent_token_embedding中操作类似)
            motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                         pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

            head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

            vel = motion_vector_a.clone() / (0.1 * self.shift)
            vel[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0
            motion_vector_a[:, (self.num_historical_steps - 1) // self.shift + 1 + t:] = 0
            x_a = torch.stack(
                [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])], dim=-1)

            x_a = self.x_a_emb(continuous_inputs=x_a.view(-1, x_a.size(-1)),
                               categorical_embs=categorical_embs)
            x_a = x_a.view(-1, num_step, self.hidden_dim)

            feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
            feat_a = self.fusion_emb(feat_a)

        agent_valid_mask[agent_category != 3] = False

        if show_detail:
            print(f"inference time per frame: {torch.mean(torch.tensor(inference_time, dtype=torch.float32)).item(): .3f}ms")

        return {
            'pos_a': pos_a[:, (self.num_historical_steps - 1) // self.shift:],
            'head_a': head_a[:, (self.num_historical_steps - 1) // self.shift:],
            'gt': data['agent']['position'][:, self.num_historical_steps:, :self.input_dim].contiguous(),
            'valid_mask': agent_valid_mask[:, self.num_historical_steps:],
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': torch.cat(next_token_idx_list, dim=-1),
            'next_token_idx_gt': agent_token_index.roll(shifts=-1, dims=1),
            'next_token_eval_mask': data['agent']['agent_valid_mask'],
            'pred_prob': pred_prob,
            'vel': vel
        }
