import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

import pdb

import math
import torch
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

from smart.model import SMART
from smart.utils.config import load_config_act
from smart.utils.log import Logging

try:
    from lxml import etree
except ImportError:
    import xml.etree.ElementTree as etree
from utils.opendrive2discretenet.opendriveparser.parser import parse_opendrive
from utils.opendrive2discretenet.network import Network
from tqdm import tqdm

np.set_printoptions(suppress=True)
SMART_DIR = os.path.dirname(__file__)
DEFAULT_ONSITE_CKPT_PATH = os.path.join(SMART_DIR, "ckpt/epoch=07-step=30440-val_loss=2.52.ckpt")
DEFAULT_MAX_TOKEN_CONTEXT = 18
DEFAULT_SAMPLING_MODE = "greedy"

class TokenizeModule:
    def __init__(self, data, map_token_path, agent_token_path):
        self.data = data

        self.argmin_sample_len = 3
        self.current_step = 10
        self.shift = 5

        map_token_traj = pickle.load(open(map_token_path, 'rb'))
        self.map_token = {'traj_src': map_token_traj['traj_src'], }
        traj_end_theta = np.arctan2(self.map_token['traj_src'][:, -1, 1] - self.map_token['traj_src'][:, -2, 1],
                                    self.map_token['traj_src'][:, -1, 0] - self.map_token['traj_src'][:, -2, 0])
        indices = torch.linspace(0, self.map_token['traj_src'].shape[1]-1, steps=self.argmin_sample_len).long()
        self.map_token['sample_pt'] = torch.from_numpy(self.map_token['traj_src'][:, indices]).to(torch.float)
        self.map_token['traj_end_theta'] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token['traj_src'] = torch.from_numpy(self.map_token['traj_src']).to(torch.float)

        agent_token_data = pickle.load(open(agent_token_path, 'rb'))
        self.trajectory_token = agent_token_data['token']
        self.trajectory_token_traj = agent_token_data['traj']
        self.trajectory_token_all = agent_token_data['token_all']
        # 对所有token依据倒数第二帧的状态为基准状态对最后一帧进行归一化
        self.token_last_all = {}
        for k, v in self.trajectory_token_all.items():
            # 计算每个 agent 的最终 token 朝向
            token_last = torch.from_numpy(v[:, -2:]).to(torch.float)    # [2048, 2, 4, 2]
            diff_xy = token_last[:, 0, 0] - token_last[:, 0, 3]         # 倒数第二帧 左前-左后
            theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])         # 倒数第二帧的航向角
            cos, sin = theta.cos(), theta.sin()
            # 生成旋转矩阵
            rot_mat = theta.new_zeros(token_last.shape[0], 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = -sin
            rot_mat[:, 1, 0] = sin
            rot_mat[:, 1, 1] = cos
            # 应用旋转矩阵并归一化 token 数据
            agent_token = torch.bmm(token_last[:, 1], rot_mat)
            agent_token -= token_last[:, 0].mean(1)[:, None, :]
            self.token_last_all[k] = agent_token.numpy()

    def tokenize(self):
        self.data = self.tokenize_map(self.data)
        self.data = self.tokenize_agent(self.data)
        return self.data

    def _wrap_angle(
            self,
            angle: torch.Tensor,
            min_val: float = -math.pi,
            max_val: float = math.pi) -> torch.Tensor:
        return min_val + (angle + max_val) % (max_val - min_val)

    def _interplating_polyline(self, polylines, heading, distance=0.5, split_distace=5):
        # 多段线切分长度为5米，多段线内部点之间距离为2.5米，即每条多段线由3个点构成
        # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
        dist_along_path_list = [[0]]
        polylines_list = [[polylines[0]]]
        for i in range(1, polylines.shape[0]):
            euclidean_dist = euclidean(polylines[i, :2], polylines[i - 1, :2])
            heading_diff = min(abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1])),
                            abs(max(heading[i], heading[i - 1]) - min(heading[1], heading[i - 1]) + math.pi))
            if heading_diff > math.pi / 4 and euclidean_dist > 3:
                dist_along_path_list.append([0])
                polylines_list.append([polylines[i]])
            elif heading_diff > math.pi / 8 and euclidean_dist > 3:
                dist_along_path_list.append([0])
                polylines_list.append([polylines[i]])
            elif heading_diff > 0.1 and euclidean_dist > 3:
                dist_along_path_list.append([0])
                polylines_list.append([polylines[i]])
            elif euclidean_dist > 10:
                dist_along_path_list.append([0])
                polylines_list.append([polylines[i]])
            else:
                dist_along_path_list[-1].append(dist_along_path_list[-1][-1] + euclidean_dist)
                polylines_list[-1].append(polylines[i])
        # plt.plot(polylines[:, 0], polylines[:, 1])
        # plt.savefig('tmp.jpg')
        new_x_list = []
        new_y_list = []
        multi_polylines_list = []
        for idx in range(len(dist_along_path_list)):
            if len(dist_along_path_list[idx]) < 2:
                continue
            dist_along_path = np.array(dist_along_path_list[idx])
            polylines_cur = np.array(polylines_list[idx])
            # Create interpolation functions for x and y coordinates
            fx = interp1d(dist_along_path, polylines_cur[:, 0])
            fy = interp1d(dist_along_path, polylines_cur[:, 1])
            # fyaw = interp1d(dist_along_path, heading)

            # Create an array of distances at which to interpolate
            new_dist_along_path = np.arange(0, dist_along_path[-1], distance)
            new_dist_along_path = np.concatenate([new_dist_along_path, dist_along_path[[-1]]])
            # Use the interpolation functions to generate new x and y coordinates
            new_x = fx(new_dist_along_path)
            new_y = fy(new_dist_along_path)
            # new_yaw = fyaw(new_dist_along_path)
            new_x_list.append(new_x)
            new_y_list.append(new_y)

            # Combine the new x and y coordinates into a single array
            new_polylines = np.vstack((new_x, new_y)).T
            polyline_size = int(split_distace / distance)
            if new_polylines.shape[0] >= (polyline_size + 1):
                padding_size = (new_polylines.shape[0] - (polyline_size + 1)) % polyline_size
                final_index = (new_polylines.shape[0] - (polyline_size + 1)) // polyline_size + 1
            else:
                padding_size = new_polylines.shape[0]
                final_index = 0
            multi_polylines = None
            new_polylines = torch.from_numpy(new_polylines)
            new_heading = torch.atan2(new_polylines[1:, 1] - new_polylines[:-1, 1],
                                    new_polylines[1:, 0] - new_polylines[:-1, 0])
            new_heading = torch.cat([new_heading, new_heading[-1:]], -1)[..., None]
            new_polylines = torch.cat([new_polylines, new_heading], -1)
            if new_polylines.shape[0] >= (polyline_size + 1):
                multi_polylines = new_polylines.unfold(dimension=0, size=polyline_size + 1, step=polyline_size)
                multi_polylines = multi_polylines.transpose(1, 2)
                multi_polylines = multi_polylines[:, ::5, :]
            if padding_size >= 3:
                last_polyline = new_polylines[final_index * polyline_size:]
                last_polyline = last_polyline[torch.linspace(0, last_polyline.shape[0] - 1, steps=3).long()]
                if multi_polylines is not None:
                    multi_polylines = torch.cat([multi_polylines, last_polyline.unsqueeze(0)], dim=0)
                else:
                    multi_polylines = last_polyline.unsqueeze(0)
            if multi_polylines is None:
                continue
            multi_polylines_list.append(multi_polylines)
        if len(multi_polylines_list) > 0:
            multi_polylines_list = torch.cat(multi_polylines_list, dim=0)
        else:
            multi_polylines_list = None
        return multi_polylines_list

    def tokenize_map(self, data):
        data['map_polygon']['type'] = data['map_polygon']['type'].to(torch.uint8)
        data['map_point']['type'] = data['map_point']['type'].to(torch.uint8)
        pt2pl = data[('map_point', 'to', 'map_polygon')]['edge_index']
        pt_type = data['map_point']['type'].to(torch.uint8)
        pt_side = torch.zeros_like(pt_type)
        pt_pos = data['map_point']['position'][:, :2]
        data['map_point']['orientation'] = self._wrap_angle(data['map_point']['orientation'])
        pt_heading = data['map_point']['orientation']
        split_polyline_type = []
        split_polyline_pos = []
        split_polyline_theta = []
        split_polyline_side = []
        pl_idx_list = []
        split_polygon_type = []
        data['map_point']['type'].unique()

        # 对多段线进行便利
        for i in sorted(np.unique(pt2pl[1])):
            # 每一条多段线对应的点
            index = pt2pl[0, pt2pl[1] == i]
            polygon_type = data['map_polygon']["type"][i]
            cur_side = pt_side[index]
            cur_type = pt_type[index]
            cur_pos = pt_pos[index]
            cur_heading = pt_heading[index]

            for side_val in np.unique(cur_side):
                for type_val in np.unique(cur_type):
                    if type_val == 13:
                        continue
                    indices = np.where((cur_side == side_val) & (cur_type == type_val))[0]
                    if len(indices) <= 2:
                        continue
                    split_polyline = self._interplating_polyline(cur_pos[indices].numpy(), cur_heading[indices].numpy())
                    if split_polyline is None:
                        continue
                    new_cur_type = cur_type[indices][0]
                    new_cur_side = cur_side[indices][0]
                    map_polygon_type = polygon_type.repeat(split_polyline.shape[0])
                    new_cur_type = new_cur_type.repeat(split_polyline.shape[0])
                    new_cur_side = new_cur_side.repeat(split_polyline.shape[0])
                    cur_pl_idx = torch.Tensor([i])
                    new_cur_pl_idx = cur_pl_idx.repeat(split_polyline.shape[0])
                    split_polyline_pos.append(split_polyline[..., :2])
                    split_polyline_theta.append(split_polyline[..., 2])
                    split_polyline_type.append(new_cur_type)
                    split_polyline_side.append(new_cur_side)
                    pl_idx_list.append(new_cur_pl_idx)
                    split_polygon_type.append(map_polygon_type)

        split_polyline_pos = torch.cat(split_polyline_pos, dim=0)
        split_polyline_theta = torch.cat(split_polyline_theta, dim=0)
        split_polyline_type = torch.cat(split_polyline_type, dim=0)
        split_polyline_side = torch.cat(split_polyline_side, dim=0)
        split_polygon_type = torch.cat(split_polygon_type, dim=0)
        pl_idx_list = torch.cat(pl_idx_list, dim=0)

        # ------------------------------------match_map_token------------------------------------

        traj_pos = split_polyline_pos.to(torch.float)
        traj_theta = split_polyline_theta[:, 0].to(torch.float)
        token_sample_pt = self.map_token['sample_pt'].to(traj_pos.device)
        token_src = self.map_token['traj_src'].to(traj_pos.device)
        max_traj_len = self.map_token['traj_src'].shape[1]
        pl_num = traj_pos.shape[0]

        # 各地图多段线的起始点坐标xy
        pt_token_pos = traj_pos[:, 0, :].clone()
        # 各地图多段线的起始位置朝向
        pt_token_orientation = traj_theta.clone()
        # 将地图多段线由全局坐标系转换为局部坐标系
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm((traj_pos - traj_pos[:, 0:1]), rot_mat.view(-1, 2, 2))
        # 将坐标转换后的多段线与地图map_token进行匹配
        distance = torch.sum((token_sample_pt[None] - traj_pos_local.unsqueeze(1))**2, dim=(-2, -1))
        pt_token_id = torch.argmin(distance, dim=1)

        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = sin
        rot_mat[..., 1, 0] = -sin
        rot_mat[..., 1, 1] = cos
        token_src_world = torch.bmm(token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
                                    rot_mat.view(-1, 2, 2)).reshape(pl_num, token_src.shape[0], max_traj_len, 2) + traj_pos[:, None, [0], :]
        token_src_world_select = token_src_world.view(-1, 1024, 11, 2)[torch.arange(pt_token_id.view(-1).shape[0]), pt_token_id.view(-1)].view(pl_num, max_traj_len, 2)

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack([torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()])
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (split_polyline_side[pt] == 0).sum()
            right_side = (split_polyline_side[pt] == 1).sum()
            center_side = (split_polyline_side[pt] == 2).sum()
            count_nums.append(torch.Tensor([left_side, right_side, center_side]))
        # count_nums: [N_polyline, 3]分别记录每个原始多段线对应的左侧、右侧、中心token有多少
        count_nums = torch.stack(count_nums, dim=0)
        # 获取每个原始多段线对应的最多token数量
        max_token_num = int(count_nums.max().item())
        # 构建多段线的轨迹掩码 [N_polyline, 3, max_token_num]
        traj_mask = torch.zeros((int(len(pl_idx_full.unique())), 3, max_token_num), dtype=bool)
        idx_matrix = torch.arange(traj_mask.size(2)).unsqueeze(0).unsqueeze(0)
        idx_matrix = idx_matrix.expand(traj_mask.size(0), traj_mask.size(1), -1)    #[N_polyline, 3, max_token_num]
        counts_num_expanded = count_nums.unsqueeze(-1)                              #[N_polyline, 3, 1]
        traj_mask[idx_matrix < counts_num_expanded] = True

        data['map_save'] = {}
        data['map_save']['traj_pos'] = split_polyline_pos
        data['map_save']['traj_theta'] = split_polyline_theta[:, 0]  # torch.arctan2(vec[:, 1], vec[:, 0])
        data['map_save']['pl_idx_list'] = pl_idx_list
        
        data['pt_token'] = {}
        data['pt_token']['type'] = split_polyline_type
        data['pt_token']['side'] = split_polyline_side
        data['pt_token']['pl_type'] = split_polygon_type
        data['pt_token']['num_nodes'] = split_polyline_pos.shape[0]
        data['pt_token']['traj_mask'] = traj_mask
        data['pt_token']['token_idx'] = pt_token_id
        data['pt_token']['position'] = torch.cat([pt_token_pos, torch.zeros((data['pt_token']['num_nodes'], 1),
                                                                            device=traj_pos.device, dtype=torch.float)], dim=-1)
        data['pt_token']['orientation'] = pt_token_orientation
        data['pt_token']['height'] = data['pt_token']['position'][:, -1]
        
        data[('pt_token', 'to', 'map_polygon')] = {}
        data[('pt_token', 'to', 'map_polygon')]['edge_index'] = token2pl
        
        return data

    def _clean_heading(self, data):
        """
            这个函数 clean_heading 的主要功能是对“heading” (朝向角度) 进行清理，以修复明显异常或突然变化的朝向角度
            （例如，当相邻帧之间的朝向差异超过一定阈值时），从而平滑朝向数据。
            具体而言，代码通过对相邻帧的朝向差异进行检测和修正，使得朝向变化更连贯。
        """
        heading = data['agent']['heading']
        valid = data['agent']['valid_mask']
        pi = torch.tensor(torch.pi)
        n_vehicles, n_frames = heading.shape

        heading_diff_raw = heading[:, :-1] - heading[:, 1:]
        heading_diff = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
        heading_diff[heading_diff > pi] -= 2 * pi
        heading_diff[heading_diff < -pi] += 2 * pi

        valid_pairs = valid[:, :-1] & valid[:, 1:]

        for i in range(n_frames - 1):
            change_needed = (torch.abs(heading_diff[:, i:i + 1]) > 1.0) & valid_pairs[:, i:i + 1]

            heading[:, i + 1][change_needed.squeeze()] = heading[:, i][change_needed.squeeze()]

            if i < n_frames - 2:
                heading_diff_raw = heading[:, i + 1] - heading[:, i + 2]
                heading_diff[:, i + 1] = torch.remainder(heading_diff_raw + pi, 2 * pi) - pi
                heading_diff[heading_diff[:, i + 1] > pi] -= 2 * pi
                heading_diff[heading_diff[:, i + 1] < -pi] += 2 * pi
        return data

    def _cal_polygon_contour(self, x, y, theta, width, length):
        """
            函数功能：计算一个矩形多边形的四个顶点坐标（轮廓）
            返回值：返回一个形状为 [n, 4, 2] 的数组 polygon_contour，表示每个矩形的四个顶点的坐标，方便后续用作绘制或碰撞检测等应用。
        """
        left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
        left_front = np.column_stack((left_front_x, left_front_y))

        right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
        right_front = np.column_stack((right_front_x, right_front_y))

        right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
        right_back = np.column_stack((right_back_x, right_back_y))

        left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
        left_back = np.column_stack((left_back_x, left_back_y))

        polygon_contour = np.concatenate(
            (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

        return polygon_contour

    def _match_token(self, pos, valid_mask, heading, category, agent_category, extra_mask):
        """
            将轨迹位置和朝向数据与预定义的 token 数据进行匹配，以便在场景中的每个时间步中都能追踪到正确的 token。
        """
        agent_num, num_step, feat_dim = pos.shape
        token_step_num = len(range(self.shift, num_step, self.shift))
        if agent_num == 0:
            return (
                torch.empty((0, token_step_num), dtype=torch.int64),
                torch.empty((0, token_step_num, 4, feat_dim), dtype=torch.float32),
            )

        agent_token_src = self.trajectory_token[category]
        token_last = self.token_last_all[category]
        if category == 'veh':
            width = 2.0
            length = 4.8
        elif category == 'cyc':
            width = 1.0
            length = 2.0
        else:
            width = 1.0
            length = 1.0

        prev_heading = heading[:, 0]
        prev_pos = pos[:, 0]
        token_num, token_contour_dim, feat_dim = agent_token_src.shape  # [2048, 4, 2]
        agent_token_src = agent_token_src.reshape(1, token_num * token_contour_dim, feat_dim).repeat(agent_num, 0)
        extra_count = int(extra_mask.sum().item())
        token_last = token_last.reshape(1, token_num * token_contour_dim, feat_dim).repeat(extra_count, 0)
        token_index_list = []
        token_contour_list = []
        prev_token_idx = None

        for i in range(self.shift, pos.shape[1], self.shift):
            # 上一token所在位置航向角（5帧前）
            theta = prev_heading
            # 当前航向角和位置
            cur_heading = heading[:, i]
            cur_pos = pos[:, i]
            # 将归一化的原始token信息以上一时刻位置和航向状态为基准调整到全局坐标系
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(agent_num, 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.from_numpy(agent_token_src).to(torch.float), rot_mat).reshape(agent_num,
                                                                                                                token_num,
                                                                                                                token_contour_dim,
                                                                                                                feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            # 获取当前所在位置的矩形四角信息
            cur_contour = self._cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            # 找出与当前距离最近的token作为匹配对象，记录该tokenid
            agent_token_index = torch.from_numpy(np.argmin(
                np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
                axis=-1))

            # 将匹配的tokenid转换为矩形四角坐标
            token_contour_select = agent_token_world[torch.arange(agent_num), agent_token_index]

            # 将当前帧信息更新为上一帧信息
            diff_xy = token_contour_select[:, 0, :] - token_contour_select[:, 3, :]
            # 数据集中原航向角
            prev_heading = heading[:, i].clone()
            # 如果是这一帧被预测的对象，则用当前token所在状态更新航向和位置信息
            prev_heading[valid_mask[:, i - self.shift]] = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])[
                valid_mask[:, i - self.shift]]

            prev_pos = pos[:, i].clone()
            prev_pos[valid_mask[:, i - self.shift]] = token_contour_select.mean(dim=1)[valid_mask[:, i - self.shift]]
            prev_token_idx = agent_token_index
            token_index_list.append(agent_token_index[:, None])
            token_contour_list.append(token_contour_select[:, None, ...])

        token_index = torch.cat(token_index_list, dim=1)
        token_contour = torch.cat(token_contour_list, dim=1)

        # extra matching（如果在第十一帧存在但第六帧不存在的代理，则根据第十帧的状态来匹配token信息）
        if extra_mask.any() and token_index.shape[1] > 1:
            theta = heading[extra_mask, self.current_step - 1]
            prev_pos = pos[extra_mask, self.current_step - 1]
            cur_pos = pos[extra_mask, self.current_step]
            cur_heading = heading[extra_mask, self.current_step]
            cos, sin = theta.cos(), theta.sin()
            rot_mat = theta.new_zeros(extra_mask.sum(), 2, 2)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            agent_token_world = torch.bmm(torch.from_numpy(token_last).to(torch.float), rot_mat).reshape(
                extra_mask.sum(), token_num, token_contour_dim, feat_dim)
            agent_token_world += prev_pos[:, None, None, :]

            cur_contour = self._cal_polygon_contour(cur_pos[:, 0], cur_pos[:, 1], cur_heading, width, length)
            agent_token_index = torch.from_numpy(np.argmin(
                np.mean(np.sqrt(np.sum((cur_contour[:, None, ...] - agent_token_world.numpy()) ** 2, axis=-1)), axis=2),
                axis=-1))
            token_contour_select = agent_token_world[torch.arange(extra_mask.sum()), agent_token_index]

            token_index[extra_mask, 1] = agent_token_index
            token_contour[extra_mask, 1] = token_contour_select

        return token_index, token_contour

    def tokenize_agent(self, data):
        # 创建插值掩码 interplote_mask，用于标记那些当前时间步为无效但坐标非零的位置，以确定需要插值的数据点
        # interplote_mask = (data['agent']['valid_mask'][:, self.current_step] == False) * (
        #         data['agent']['position'][:, self.current_step, 0] != 0)
        interplote_mask = (data['agent']['valid_mask'][:, self.current_step] == True) * (data['agent']['valid_mask'][:, self.current_step-1] == False)
        # 通过检查当前时间步中无效但位置非零的轨迹点，将其前一个时间步的位置、速度、航向等信息进行估算和填充，确保轨迹数据连续性
        if data['agent']["velocity"].shape[-1] == 2:
            data['agent']["velocity"] = torch.cat([data['agent']["velocity"],
                                                    torch.zeros(data['agent']["velocity"].shape[0],
                                                                data['agent']["velocity"].shape[1], 1)], dim=-1)
        vel = data['agent']["velocity"][interplote_mask, self.current_step]
        # 插值前一个时间步的位置、航向、速度
        data['agent']['position'][interplote_mask, self.current_step - 1, :3] = data['agent']['position'][
                                                                                interplote_mask, self.current_step,
                                                                                :3] - vel * 0.1
        data['agent']['heading'][interplote_mask, self.current_step - 1] = data['agent']['heading'][
            interplote_mask, self.current_step]
        data['agent']["velocity"][interplote_mask, self.current_step - 1] = data['agent']["velocity"][
            interplote_mask, self.current_step]
        data['agent']['valid_mask'][interplote_mask, self.current_step - 1:self.current_step + 1] = True

        data['agent']['type'] = data['agent']['type'].to(torch.uint8)

        data = self._clean_heading(data)
        matching_extra_mask = (data['agent']['valid_mask'][:, self.current_step] == True) * (
                data['agent']['valid_mask'][:, self.current_step - 5] == False)

        interplote_mask_first = (data['agent']['valid_mask'][:, 0] == False) * (data['agent']['position'][:, 0, 0] != 0)
        data['agent']['valid_mask'][interplote_mask_first, 0] = True

        agent_pos = data['agent']['position'][:, :, :2]
        valid_mask = data['agent']['valid_mask']
        # 以下标1为起点，长度为6，间隔为5创建滑动窗口
        valid_mask_shift = valid_mask.unfold(1, self.shift + 1, self.shift)         # [NA, 18, 6]
        # 每个滑动窗口的起止都为true时窗口才有效
        token_valid_mask = valid_mask_shift[:, :, 0] * valid_mask_shift[:, :, -1]   # [NA, 18]
        agent_type = data['agent']['type']
        agent_category = data['agent']['category']
        agent_heading = data['agent']['heading']
        vehicle_mask = agent_type == 0
        cyclist_mask = agent_type == 2
        ped_mask = agent_type == 1

        veh_pos = agent_pos[vehicle_mask, :, :]
        veh_valid_mask = valid_mask[vehicle_mask, :]
        cyc_pos = agent_pos[cyclist_mask, :, :]
        cyc_valid_mask = valid_mask[cyclist_mask, :]
        ped_pos = agent_pos[ped_mask, :, :]
        ped_valid_mask = valid_mask[ped_mask, :]

        veh_token_index, veh_token_contour = self._match_token(veh_pos, veh_valid_mask, agent_heading[vehicle_mask],
                                                                'veh', agent_category[vehicle_mask],
                                                                matching_extra_mask[vehicle_mask])
        ped_token_index, ped_token_contour = self._match_token(ped_pos, ped_valid_mask, agent_heading[ped_mask], 'ped',
                                                                agent_category[ped_mask], matching_extra_mask[ped_mask])
        cyc_token_index, cyc_token_contour = self._match_token(cyc_pos, cyc_valid_mask, agent_heading[cyclist_mask],
                                                                'cyc', agent_category[cyclist_mask],
                                                                matching_extra_mask[cyclist_mask])

        # token_index: [NA, 18(90/5)] 每个代理在90帧中匹配到的18个token索引
        token_index = torch.zeros((agent_pos.shape[0], veh_token_index.shape[1])).to(torch.int64)
        token_index[vehicle_mask] = veh_token_index
        token_index[ped_mask] = ped_token_index
        token_index[cyclist_mask] = cyc_token_index

        # token_contour: [NA, 18, 4, 2] 每个代理在90帧中匹配到的18个token对应的矩形信息
        token_contour = torch.zeros((agent_pos.shape[0], veh_token_contour.shape[1],
                                        veh_token_contour.shape[2], veh_token_contour.shape[3]))
        token_contour[vehicle_mask] = veh_token_contour
        token_contour[ped_mask] = ped_token_contour
        token_contour[cyclist_mask] = cyc_token_contour

        token_valid_mask[matching_extra_mask, 1] = True

        data['agent']['token_idx'] = token_index            # [NA, 18]
        data['agent']['token_contour'] = token_contour      # [NA, 18, 4, 2]
        token_pos = token_contour.mean(dim=2)               
        data['agent']['token_pos'] = token_pos              # [NA, 18, 2]
        diff_xy = token_contour[:, :, 0, :] - token_contour[:, :, 3, :]
        data['agent']['token_heading'] = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])  # [NA, 18]
        data['agent']['agent_valid_mask'] = token_valid_mask                                # [NA, 18]

        vel = torch.cat([token_pos.new_zeros(data['agent']['num_nodes'], 1, 2),
                            ((token_pos[:, 1:] - token_pos[:, :-1]) / (0.1 * self.shift))], dim=1)
        vel_valid_mask = torch.cat([torch.zeros(token_valid_mask.shape[0], 1, dtype=torch.bool),
                                    (token_valid_mask * token_valid_mask.roll(shifts=1, dims=1))[:, 1:]], dim=1)
        vel[~vel_valid_mask] = 0
        vel[data['agent']['valid_mask'][:, self.current_step], 1] = data['agent']['velocity'][
                                                                    data['agent']['valid_mask'][:, self.current_step],
                                                                    self.current_step, :2]

        data['agent']['token_velocity'] = vel

        return data


def load_onsite_smart_model(ckpt_path=None):
    config = load_config_act(os.path.join(SMART_DIR, "configs/validation/validation_scalable.yaml"))
    model = SMART(config.Model)
    if ckpt_path is None:
        ckpt_path = DEFAULT_ONSITE_CKPT_PATH
    ckpt_path = os.fspath(ckpt_path)
    model.load_params_from_file(filename=ckpt_path, logger=Logging().log(level='DEBUG'))
    model.eval()
    return model


class OnSiteSmartAgent:
    def __init__(
        self,
        data,
        model=None,
        max_token_context=DEFAULT_MAX_TOKEN_CONTEXT,
        sampling_mode=DEFAULT_SAMPLING_MODE,
    ):
        self.map_token_traj_path = os.path.join(SMART_DIR, "smart/tokens/map_traj_token5.pkl")
        self.agent_token_path = os.path.join(SMART_DIR, "smart/tokens/cluster_frame_5_2048.pkl")
        self.tokenizer = TokenizeModule(data, map_token_path=self.map_token_traj_path, agent_token_path=self.agent_token_path)
        self.data = self.tokenizer.tokenize()
        self.max_token_context = max_token_context
        if sampling_mode not in {"greedy", "topk_sample"}:
            raise ValueError(f"unsupported sampling_mode: {sampling_mode}")
        self.sampling_mode = sampling_mode

        self.model = model if model is not None else load_onsite_smart_model()
        self.model.eval()

        self.trajectory_token_all_veh = torch.from_numpy(self.tokenizer.trajectory_token_all['veh']).clone().to(torch.float)
        self.trajectory_token_all_ped = torch.from_numpy(self.tokenizer.trajectory_token_all['ped']).clone().to(torch.float)
        self.trajectory_token_all_cyc = torch.from_numpy(self.tokenizer.trajectory_token_all['cyc']).clone().to(torch.float)

        self.map_encoder = self.model.encoder.map_encoder
        self.map_emb = self.map_encoder.inference(self.data).detach()

        self.agent_encoder = self.model.encoder.agent_encoder
        self.map_range = [[self.data['map_point']['position'][:, 0].min() - 20, self.data['map_point']['position'][:, 0].max() + 20],
                          [self.data['map_point']['position'][:, 1].min() - 20, self.data['map_point']['position'][:, 1].max() + 20]]


    def inference(self, total_frame=120, progress_callback=None):
        while self.data['agent']['position'].shape[1] < total_frame:
            if progress_callback is not None:
                progress_callback(self.data['agent']['position'].shape[1], total_frame)
            self.next_inference()

        return {
            'valid': self.data['agent']['valid_mask'][:, :total_frame],
            'position': self.data['agent']['position'][:, :total_frame, :2],
            'heading': self.data['agent']['heading'][:, :total_frame],
            'velocity': torch.norm(self.data['agent']['velocity'], p=2, dim=-1)
        }


    def next_inference(self):
        if self.max_token_context is not None and self.max_token_context > 0:
            context = slice(max(0, self.data['agent']['token_pos'].shape[1] - self.max_token_context), None)
        else:
            context = slice(None)
        pos_a = self.data['agent']['token_pos'][:, context].clone()
        head_a = self.data['agent']['token_heading'][:, context].clone()
        mask = self.data['agent']['agent_valid_mask'][:, context].clone()
        type_a = self.data["agent"]["type"]
        agent_token_index = self.data['agent']['token_idx'][:, context]
        agent_category = self.data['agent']['category'] 

        num_agent, num_step, traj_dim = pos_a.shape
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        
        # feat_a 表示对agent信息进行编码后的向量 [NA, 18, 128]
        # - 融合 轨迹token、代理类型、代理形状、代理行驶特征（token间距离和航向角变化） 进行词嵌入
        # agent_token_traj 表示记录了每个token对应的轨迹信息 [NA, 18, 2048, 4, 2]
        feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs = self.agent_encoder.agent_token_embedding(
            self.data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True)
        
        # edge_index_t ([2, E_t]) 同一代理在时间窗time_span内的不同有效时刻间建立边
        # r_t ([E_t, 128]) 对同一代理在时间窗time_span内的不同有效时刻间信息进行embedding
        # - 融合 同一代理不同时刻间的距离、当前航向与位移向量之间夹角、同一代理不同时刻间的航向角之差、token所在时刻的索引之差 进行词嵌入
        edge_index_t, r_t = self.agent_encoder.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        batch_s = torch.arange(num_step, device=pos_a.device).repeat_interleave(num_agent)
        batch_pl = torch.arange(num_step, device=pos_a.device).repeat_interleave(self.data['pt_token']['num_nodes'])
        mask_s = mask.transpose(0, 1).reshape(-1)     # [NA * 18]
        # edge_index_a2a ([2, E_a2a])  同一时刻位置在60米范围内的关联代理之间建立边
        # r_a2a ([E_a2a, 128]) 对同一时刻不同关联代理之间的信息进行embedding
        # - 融合 同一时刻关联代理之间的距离、航向差、位移向量与关联代理1航向的夹角 进行词嵌入
        edge_index_a2a, r_a2a = self.agent_encoder.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)
        
        # 筛选出被预测代理
        mask[self.data['agent']['category'] != 3] = False
        # edge_index_pl2a ([2, E_pl2a]) 同一时刻代理与多段线之间的有效边
        # r_pl2a ([E_pl2a, 128]) 对同一时刻代理与关联多段线间的信息进行embedding
        # - 融合 同一时刻代理到多段线的距离、代理航向向量 转到 代理指向多段线坐标的位移向量 的角度、代理航向角转到多段线朝向的有向角度 进行词嵌入
        edge_index_pl2a, r_pl2a = self.agent_encoder.build_map2agent_edge(self.data, num_step, pos_a, head_a,
                                                            head_vector_a, mask, batch_s, batch_pl)

        for i in range(self.agent_encoder.num_layers):
            # 同一代理不同时刻之间进行自注意力机制
            feat_a = feat_a.reshape(-1, self.agent_encoder.hidden_dim)                # [NA, 18, 12] -> [18 * NA, 128]
            feat_a = self.agent_encoder.t_attn_layers[i](feat_a, r_t, edge_index_t)   # [18 * NA, 128]
            
            # 同一时刻代理与关联多段线之间进行交叉注意力机制
            feat_a = feat_a.reshape(-1, num_step,
                                    self.agent_encoder.hidden_dim).transpose(0, 1).reshape(-1, self.agent_encoder.hidden_dim)   # [18 * NA, 128] -> [NA * 18, 128]
            feat_a = self.agent_encoder.pt2a_attn_layers[i]((self.map_emb.repeat_interleave(
                repeats=num_step, dim=0).reshape(-1, num_step, self.agent_encoder.hidden_dim).transpose(0, 1).reshape(
                    -1, self.agent_encoder.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            
            # 同一时刻不同关联代理之间进行自注意力机制
            feat_a = self.agent_encoder.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)         # [NA * 18, 128]
            feat_a = feat_a.reshape(num_step, -1, self.agent_encoder.hidden_dim).transpose(0, 1)  # [NA, 18, 128]

        self.agent_encoder.beam_size = 10
        next_token_prob = self.agent_encoder.token_predict_head(feat_a[:, -1])
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        topk_prob, next_token_idx = torch.topk(next_token_prob_softmax, k=self.agent_encoder.beam_size, dim=-1)

        expanded_index = next_token_idx[..., None, None, None].expand(-1, -1, 6, 4, 2)
        next_token_traj = torch.gather(agent_token_traj_all, 1, expanded_index)                  # [NA, 5, 6, 4, 2]
        theta = head_a[:, -1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = torch.zeros((num_agent, 2, 2), device=theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        # 将token轨迹转换到全局坐标系
        agent_diff_rel = torch.bmm(next_token_traj.view(-1, 4, 2),
                                    rot_mat[:, None, None, ...].repeat(1, self.agent_encoder.beam_size, self.agent_encoder.shift + 1, 1, 1).view(
                                        -1, 2, 2)).view(num_agent, self.agent_encoder.beam_size, self.agent_encoder.shift + 1, 4, 2)
        agent_pred_rel = agent_diff_rel + pos_a[:, -1, :][:, None, None, None, ...]  # [NA, 5, 6, 4, 2]

        if self.sampling_mode == "greedy":
            sample_index = torch.zeros((num_agent, 1), dtype=torch.long, device=agent_pred_rel.device)
        else:
            sample_index = torch.multinomial(topk_prob, 1).to(agent_pred_rel.device)                                # [NA, 1] 根据topk_prob进行采样
        agent_pred_rel = agent_pred_rel.gather(dim=1,
                                                index=sample_index[..., None, None, None].expand(-1, -1, 6, 4,
                                                                                                2))[:, 0, ...]      # [NA, 6, 4, 2] 根据采样结果得到token的全局轨迹
        pred_prob = topk_prob.gather(dim=-1, index=sample_index)[:, 0]
        pred_traj = agent_pred_rel.clone().mean(dim=2)
        diff_xy = agent_pred_rel[:, 1:, 0, :] - agent_pred_rel[:, 1:, 3, :]
        pred_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])     # [NA, 5] 更新未来5帧的预测航向角

        x_valid = (pred_traj[:, 1:, 0] >= self.map_range[0][0]) & (pred_traj[:, 1:, 0] <= self.map_range[0][1])
        y_valid = (pred_traj[:, 1:, 1] >= self.map_range[1][0]) & (pred_traj[:, 1:, 1] <= self.map_range[1][1])
        valid = x_valid & y_valid
        # valid = torch.ones((num_agent, self.agent_encoder.shift), dtype=torch.bool)
        self.data['agent']['valid_mask'] = torch.cat([self.data['agent']['valid_mask'], valid], dim=1)
        self.data['agent']['position'] = torch.cat([self.data['agent']['position'], torch.cat([pred_traj[:, 1:], torch.zeros((num_agent, self.agent_encoder.shift, 1), dtype=torch.float32)], dim=-1)], dim=1)
        self.data['agent']['heading'] = torch.cat([self.data['agent']['heading'], pred_head], dim=1)
        velocity = torch.cat([(pred_traj[:, 1:] - pred_traj[:, :-1]) / 0.1, torch.zeros((num_agent, self.agent_encoder.shift, 1), dtype=torch.float32)], dim=-1)
        self.data['agent']['velocity'] = torch.cat([self.data['agent']['velocity'], velocity], dim=1)
        self.data['agent']['shape'] = torch.cat([self.data['agent']['shape'], self.data['agent']['shape'][:, :self.agent_encoder.shift]], dim=1)
        
        next_token_idx = next_token_idx.gather(dim=1, index=sample_index)
        self.data['agent']['token_idx'] = torch.cat([self.data['agent']['token_idx'], next_token_idx], dim=1)
        self.data['agent']['token_contour'] = torch.cat([self.data['agent']['token_contour'], agent_pred_rel[:, -1:, ...]], dim=1)
        self.data['agent']['token_pos'] = torch.cat([self.data['agent']['token_pos'], pred_traj[:, -1:]], dim=1)
        diff_xy = agent_pred_rel[:, -1, 0, :] - agent_pred_rel[:, -1, 3, :]
        theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
        self.data['agent']['token_heading'] = torch.cat([self.data['agent']['token_heading'], theta[:, None]], dim=1)
        self.data['agent']['agent_valid_mask'] = torch.cat([self.data['agent']['agent_valid_mask'], torch.all(valid, dim=1, keepdim=True)], dim=1)
        token_velocity = (pred_traj[:, -1] - pred_traj[:, 0]) / (0.1 * self.agent_encoder.shift)
        self.data['agent']['token_velocity'] = torch.cat([self.data['agent']['token_velocity'], token_velocity[:, None]], dim=1)

    def visualize(self, save_path=None):
        ego_index = self.data['agent']['av_index']
        valid = self.data['agent']['valid_mask']
        traj = self.data['agent']['position'][..., :2]
        head = self.data['agent']['heading']

        N, T, _ = traj.shape
        agent_traj_all = self._cal_polygon_contour(
            traj.view(-1, 2)[..., 0], 
            traj.view(-1, 2)[..., 1], 
            head.view(-1), 
            self.data['agent']['shape'].view(-1, 3)[..., 1], 
            self.data['agent']['shape'].view(-1, 3)[..., 0]
        ).reshape(N, T, 4, 2)

        fig, ax_map = plt.subplots(figsize=(20, 20))
        ax_agent = ax_map.twinx()
        self.plot_static_map(ax_map, self.data)

        def update(frame):
            ax_agent.cla()
            ax_agent.axis('off')
            ax_agent.set_ylim(ax_map.get_ylim())
            polygons = []
            for agent_idx in range(agent_traj_all.shape[0]):
                if valid[agent_idx, frame]:
                    fill_color = 'red' if agent_idx == ego_index else 'blue'
                    polygon = patches.Polygon(agent_traj_all[agent_idx, frame], closed=True, fill=True, facecolor=fill_color, edgecolor=None, linewidth=2, alpha=0.9)  # fill=None 使其不填充
                    ax_agent.add_patch(polygon)
                    polygons.append(polygon)
            return polygons

        ani = FuncAnimation(fig, update, frames=np.arange(T), blit=True)

        if save_path:
            ani.save(save_path, writer=PillowWriter(fps=10))
        else:
            plt.show()

    def plot_static_map(self, ax, batch):
        # 0:'DASH_SOLID_YELLOW', 1:'DASH_SOLID_WHITE', 2:'DASHED_WHITE', 3:'DASHED_YELLOW', 4:'DOUBLE_SOLID_YELLOW', 5:'DOUBLE_SOLID_WHITE', 6:'DOUBLE_DASH_YELLOW', 7:'DOUBLE_DASH_WHITE',
        # 8:'SOLID_YELLOW', 9:'SOLID_WHITE', 10:'SOLID_DASH_WHITE', 11:'SOLID_DASH_YELLOW', 12:'EDGE', 13:'NONE', 14:'UNKNOWN', 15:'CROSSWALK', 16:'CENTERLINE'
        _line_style = [['--', 2, 'yellow'], ['--', 2, 'grey'], ['--', 2, 'grey'], ['--', 2, 'yellow'], ['-', 2, 'yellow'], ['-', 2, 'grey'], ['--', 2, 'yellow'], ['--', 2, 'grey'],
                    ['-', 2, 'yellow'], ['-', 2, 'grey'], ['--', 2, 'grey'], ['--', 2, 'yellow'], ['-', 3, 'black'], [], [], [':', 2, 'blue'], []]
        _center_colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightgray']

        # 准备数据
        polylines = []
        polyline_type = []
        for i in range(batch['map_polygon']['num_nodes']):
            point_idx = batch[('map_point', 'to', 'map_polygon')]['edge_index'][0, batch[('map_point', 'to', 'map_polygon')]['edge_index'][1] == i]
            polylines.append(torch.gather(batch['map_point']['position'][:, :2], dim=0, index=point_idx[..., None].repeat(1, 2)))
            polyline_type.append(batch['map_point']['type'][point_idx[0]])

        # 绘制每条地图线段
        for idx, (type, data) in enumerate(zip(polyline_type, polylines)):
            x = data[:, 0].numpy()
            y = data[:, 1].numpy()
            if (type == 13 or type == 14):
                continue
            elif (type == 16):
                ax.plot(x, y, marker='', linestyle='-', linewidth=2, color=_center_colors[batch['map_polygon']['light_type'][idx]], alpha=0.5)
            else:
                ax.plot(x, y, marker='', linestyle=_line_style[type][0], linewidth=_line_style[type][1], color=_line_style[type][2], alpha=0.8)
        range_x, range_y = self._cal_proportional_range([batch['map_point']['position'][:, 0].min(), batch['map_point']['position'][:, 0].max()], 
                                [batch['map_point']['position'][:, 1].min(), batch['map_point']['position'][:, 1].max()], 1.0)
        ax.set_xlim(*range_x)
        ax.set_ylim(*range_y)
        ax.set_aspect('equal')
        ax.set_title(f"Scene <{batch['scenario_id']}>")

    @staticmethod
    def _cal_proportional_range(range_x, range_y, aspect_ratio):
        """计算比例范围
        Args:
            range_x (list): x轴范围
            range_y (list): y轴范围
            aspect_ratio (float): 长宽比
        Returns:
            list: 比例范围
        """
        len_x = range_x[1] - range_x[0]
        len_y = range_y[1] - range_y[0]
        center = [(range_x[0] + range_x[1]) / 2, (range_y[0] + range_y[1]) / 2]

        if len_x > len_y * aspect_ratio:
            len_y = len_x / aspect_ratio
        else:
            len_x = len_y * aspect_ratio
        return [center[0] - len_x / 2, center[0] + len_x / 2], [center[1] - len_y / 2, center[1] + len_y / 2]

    @staticmethod
    def _cal_polygon_contour(x, y, theta, width, length):
        left_front_x = x + 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        left_front_y = y + 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
        left_front = np.column_stack((left_front_x, left_front_y))

        right_front_x = x + 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        right_front_y = y + 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
        right_front = np.column_stack((right_front_x, right_front_y))

        right_back_x = x - 0.5 * length * np.cos(theta) + 0.5 * width * np.sin(theta)
        right_back_y = y - 0.5 * length * np.sin(theta) - 0.5 * width * np.cos(theta)
        right_back = np.column_stack((right_back_x, right_back_y))

        left_back_x = x - 0.5 * length * np.cos(theta) - 0.5 * width * np.sin(theta)
        left_back_y = y - 0.5 * length * np.sin(theta) + 0.5 * width * np.cos(theta)
        left_back = np.column_stack((left_back_x, left_back_y))

        polygon_contour = np.concatenate(
            (left_front[:, None, :], right_front[:, None, :], right_back[:, None, :], left_back[:, None, :]), axis=1)

        return polygon_contour


def get_xodr_info(xodr_path: str):
    with open(xodr_path, 'r', encoding='utf-8') as fh:
        root = etree.parse(fh).getroot()
    openDriveXml = parse_opendrive(root)
    loadedRoadNetwork = Network()
    loadedRoadNetwork.load_opendrive(openDriveXml)

    vehicle_road_types = ["driving","biking", "onRamp", "offRamp", "exit", "entry", "sidewalk", "bidirectional"]
    bicycle_road_types = ["biking"]
    pedestrain_road_types = ["sidewalk", "crosswalk"]
    # 获取地图中处在交叉口内的lane
    juntion_lanes = []
    # 获取地图中处在车道最左侧的lane
    left_bound_lanes = []
    # 获取地图中处在车道最右侧的lane
    right_bound_lanes = []
    roads = root.findall("road")
    for road in roads:
        road_id = road.get("id")
        junction_id = road.get("junction")

        lane_sections = road.findall(".//laneSection")
        for section_id, section in enumerate(lane_sections):
            lanes = section.findall("left/lane")
            if junction_id == '-1':
                for lane in lanes:
                    if lane.get('type') in vehicle_road_types:
                        lane_id = lane.get('id')
                        right_bound_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
                        break
                for lane in lanes[::-1]:
                    if lane.get('type') in vehicle_road_types:
                        lane_id = lane.get('id')
                        left_bound_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
                        break
            else:
                for lane in lanes:
                    lane_id = lane.get('id')
                    juntion_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
            lanes = section.findall("right/lane")
            if junction_id == '-1':
                for lane in lanes:
                    if lane.get('type') in vehicle_road_types:
                        lane_id = lane.get('id')
                        left_bound_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
                        break
                for lane in lanes[::-1]:
                    if lane.get('type') in vehicle_road_types:
                        lane_id = lane.get('id')
                        right_bound_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
                        break
            else:
                for lane in lanes:
                    lane_id = lane.get('id')
                    juntion_lanes.append(f"{road_id}.{section_id}.{lane_id}.-1")
    
    polygons = []
    point_types = []
    polygon_types = []
    polygon_light_types = []
    _point_types = ['DASH_SOLID_YELLOW', 'DASH_SOLID_WHITE', 'DASHED_WHITE', 'DASHED_YELLOW',
                    'DOUBLE_SOLID_YELLOW', 'DOUBLE_SOLID_WHITE', 'DOUBLE_DASH_YELLOW', 'DOUBLE_DASH_WHITE',
                    'SOLID_YELLOW', 'SOLID_WHITE', 'SOLID_DASH_WHITE', 'SOLID_DASH_YELLOW', 'EDGE',
                    'NONE', 'UNKNOWN', 'CROSSWALK', 'CENTERLINE']
    _polygon_types = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
    _polygon_light_type = ['LANE_STATE_STOP', 'LANE_STATE_GO', 'LANE_STATE_CAUTION', 'LANE_STATE_UNKNOWN']
    openDriveXml = parse_opendrive(root)
    loadedRoadNetwork = Network()
    loadedRoadNetwork.load_opendrive(openDriveXml)
    def add_lane_info(points, point_type, polygon_type, polygon_light_type):
        polygons.append(torch.cat((torch.tensor(points, dtype=torch.float32), torch.zeros(len(points), 1, dtype=torch.float32)), dim=1))
        point_types.append(_point_types.index(point_type))
        polygon_types.append(_polygon_types.index(polygon_type))
        polygon_light_types.append(_polygon_light_type.index(polygon_light_type))
    # 解析机动车道信息
    vehicle_map_info = loadedRoadNetwork.export_discrete_network(filter_types=vehicle_road_types)
    for lane_info in vehicle_map_info.discretelanes:
        add_lane_info(lane_info.center_vertices, 'CENTERLINE', 'VEHICLE', 'LANE_STATE_UNKNOWN')

        if lane_info.lane_id in right_bound_lanes:
            add_lane_info(lane_info.right_vertices, 'EDGE', 'VEHICLE', 'LANE_STATE_UNKNOWN')
        elif lane_info.lane_id not in juntion_lanes:
            add_lane_info(lane_info.right_vertices, 'DASHED_WHITE', 'VEHICLE', 'LANE_STATE_UNKNOWN')
        
        if lane_info.lane_id in left_bound_lanes:
            add_lane_info(lane_info.left_vertices, 'SOLID_WHITE', 'VEHICLE', 'LANE_STATE_UNKNOWN')
    # 解析非机动车道信息
    bicycle_map_info = loadedRoadNetwork.export_discrete_network(filter_types=bicycle_road_types)
    for lane_info in bicycle_map_info.discretelanes:
        add_lane_info(lane_info.center_vertices, 'CENTERLINE', 'BIKE', 'LANE_STATE_UNKNOWN')
    # 解析人行道信息
    pedestrain_map_info = loadedRoadNetwork.export_discrete_network(filter_types=pedestrain_road_types)
    for lane_info in pedestrain_map_info.discretelanes:
        add_lane_info(lane_info.center_vertices, 'CROSSWALK', 'PEDESTRIAN', 'LANE_STATE_UNKNOWN')
    
    points = []
    orientations = []
    magnitudes =[]
    for polyline in polygons:
        point = polyline[:-1]
        center_vectors = polyline[1:] - polyline[:-1]
        point_orientation = torch.cat([torch.atan2(center_vectors[:, 1], center_vectors[:, 0])], dim=0)
        point_magnitude = torch.norm(torch.cat([center_vectors[:, :2]], dim=0), p=2, dim=-1)
        points.append(point)
        orientations.append(point_orientation)
        magnitudes.append(point_magnitude)
    pl_idx = torch.cat([torch.ones(p.shape[0]) * i for i, p in enumerate(points)]).to(torch.int64)
    point_types = torch.cat([torch.ones(p.shape[0]) * point_types[i] for i, p in enumerate(points)]).to(torch.int8)
    points = torch.cat(points, dim=0)
    orientations = torch.cat(orientations, dim=0)
    magnitudes = torch.cat(magnitudes, dim=0)
    mp2ml = torch.stack([torch.arange(points.shape[0]), pl_idx], dim=0).to(dtype=torch.int64)

    map_info = {
        # "scenario_id": scene_data['scene_name'],
        "map_polygon": {
            "num_nodes": len(polygons),
            "type": torch.tensor(polygon_types, dtype=torch.uint8),
            "light_type": torch.tensor(polygon_light_types, dtype=torch.uint8),
        },
        "map_point": {
            "num_nodes": points.shape[0],
            "position": points,
            "orientation": orientations,
            "magnitude": magnitudes,
            "height": points[:, 2],
            "type": point_types,
        },
        ('map_point', 'to', 'map_polygon'): {
            "edge_index": mp2ml
        }
    }
    return map_info

def compute_velocities(positions: torch.Tensor, valid_mask: torch.Tensor, dt: float = 0.1):
    """
    计算矢量速度，满足以下条件：
    1. 速度计算为 v_t = (p_t - p_{t-1}) / dt
    2. 仅在相邻帧都有效时计算速度，否则保持0
    3. 每个代理的第一个有效时刻的速度等于第二个有效时刻的速度

    Args:
        positions (torch.Tensor): 形状 (B, T, D)，表示 B 个代理 T 帧的 3D 位置。
        valid_mask (torch.Tensor): 形状 (B, T)，表示每个代理每帧是否有效。
        dt (float): 时间间隔，默认为 0.1。

    Returns:
        torch.Tensor: 形状 (B, T, D) 的矢量速度。
    """
    B, T, D = positions.shape

    # 计算相邻帧的速度（默认值为0）
    velocities = torch.zeros_like(positions)
    diffs = (positions[:, 1:] - positions[:, :-1]) / dt  # 计算相邻帧的位移变化除以时间间隔

    # 只有在当前帧和前一帧都有效的情况下，速度才有效
    valid_prev = valid_mask[:, :-1]  # t-1 的有效性
    valid_curr = valid_mask[:, 1:]   # t 的有效性
    valid_velocity = valid_prev & valid_curr  # 只有连续有效帧才能计算速度

    # 更新速度张量（跳过第0帧）
    velocities[:, 1:][valid_velocity] = diffs[valid_velocity]

    # 处理每个代理的第一个有效时刻
    for i in range(B):
        valid_indices = torch.where(valid_mask[i])[0]  # 找到代理 i 的有效时间索引
        if len(valid_indices) > 1:  # 需要至少2个有效帧才能设置初始速度
            first_valid, second_valid = valid_indices[:2]
            velocities[i, first_valid] = velocities[i, second_valid]  # 赋值第一个有效帧的速度
    
    return velocities

def get_pkl_info(pkl_path: str, history_frames: int=31):
    with open(pkl_path, 'rb') as handle:
        scene_data = pickle.load(handle)

    num_veh = len(scene_data['ids'])
    av_index = [scene_data['ids'].index('Ego')] if 'Ego' in scene_data['ids'] else []
    categories = torch.ones(num_veh, dtype=torch.uint8) * 3
    shapes = torch.cat([torch.tensor(scene_data['shapes']), torch.zeros((num_veh, 1), dtype=torch.float32)], dim=1)[:, None, :].repeat(1, history_frames, 1)
    positions = torch.cat([torch.tensor(scene_data['positions'][:, :history_frames]), torch.zeros((num_veh, history_frames, 1), dtype=torch.float32)], dim=-1)
    headings = torch.tensor(scene_data['headings'][:, :history_frames])
    velocities = compute_velocities(torch.tensor(scene_data['positions']), torch.tensor(scene_data['valid_mask']))
    velocities = torch.cat([velocities[:, :history_frames], torch.zeros((num_veh, history_frames, 1), dtype=torch.float32)], dim=-1)

    agent_info = {
        'num_nodes': num_veh,
        'av_index': av_index,
        'valid_mask': torch.tensor(scene_data['valid_mask'][:, :history_frames]),
        'predict_mask': torch.tensor(scene_data['predict_mask']),
        'id': scene_data['ids'],
        'type': torch.tensor(scene_data['types']),
        'category': categories,
        'position': positions,
        'heading': headings,
        'velocity': velocities,
        'shape': shapes,
    }
    return agent_info

def main(scenarios: str):
    # 遍历所有场景文件夹
    for scenario_name in tqdm(os.listdir(scenarios), desc="Processing Scenarios"):
        if scenario_name != '0004follow5':
            continue

        scenario_dir = os.path.join(scenarios, scenario_name)

        if not os.path.isdir(scenario_dir):
            continue

        # 加载XODR和PKL路径
        xodr_path = os.path.join(scenario_dir, f"{scenario_name}.xodr")
        assert os.path.exists(xodr_path), "XODR文件不存在"

        pkl_path = os.path.join(scenario_dir, f"{scenario_name}_exam.pkl")
        assert os.path.exists(pkl_path), "PKL文件不存在"

        # 加载场景数据
        with open(pkl_path, 'rb') as handle:
            scene_data = pickle.load(handle)

        # 获取地图和智能体信息
        map_info = get_xodr_info(xodr_path)
        agent_info = get_pkl_info(pkl_path)
        scenario_info = {
            'scenario_id': scenario_name,
            **map_info,
            'agent': agent_info
        }
        
        # 初始化在线推理模型并进行推理
        onsite_model = OnSiteSmartAgent(scenario_info)
        inference_info = onsite_model.inference(total_frame=scene_data['sim_duration'])

        # 更新场景数据
        scene_data['valid_mask'] = inference_info['valid'].numpy()
        scene_data['positions'] = inference_info['position'].numpy()
        scene_data['headings'] = inference_info['heading'].numpy()

        for veh_num in range(scene_data['valid_mask'].shape[0]):
            valid_position = scene_data['positions'][veh_num][scene_data['valid_mask'][veh_num]]
            print(valid_position.shape)
            
            # 计算相邻帧之间的位移距离
            dt = 0.1
            velocity = np.sqrt(np.sum(np.diff(valid_position, axis=0)**2, axis=1)) / dt
            print(velocity.shape)
            acceleration = np.diff(velocity) / dt
            print(acceleration.shape)

            for i, (v, a) in enumerate(zip(velocity, acceleration)):
                print(i, v, a)

        # 保存推理结果
        output_pkl_path = f"/home/yangyh408/codes/SMART/outputs/scenario_v3/{scenario_name}_output.pkl"
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(scene_data, f)

        # 可视化并保存GIF
        output_gif_dir = f"/home/yangyh408/codes/SMART/outputs/scenario_v3/{scenario_name}.gif"
        onsite_model.visualize(save_path=output_gif_dir)

        break


if __name__ == '__main__':
    scenarios = r"/home/yangyh408/codes/onsite-generate-scene-test-dev/ground_truth"
    main(scenarios)
    # with open(os.path.join(SMART_DIR, "data/onsite/intersection_12_61_4.pkl"), 'rb') as handle:
    #     data = pickle.load(handle)
    # onsite_model = OnSiteSmartAgent(data)
    # inference_info = onsite_model.inference(total_frame=61)
    # onsite_model.visualize()
