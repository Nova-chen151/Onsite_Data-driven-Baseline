<div align="center">
<a href="https://onsite.com.cn/">
    <img src="asset/ONSITE-blue-logo-cn_name.svg" alt="OnSite" width="800">
</a>

# OnSite 训测一体场景生成赛道基线：面向自动驾驶训测的数据驱动仿真场景生成

<p>
  <a href="README.md">English</a> · <strong>中文</strong>
</p>
</div>

<div align="center">
<a href="https://onsite.com.cn/"><img src="https://img.shields.io/badge/OnSite-4.0-blue"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://tops.tongji.edu.cn/"><img src="https://img.shields.io/badge/TCU-TOPS-purple"></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Python-3.11-yellow">
</div>

## 基本概述

本项目用于 OnSite 训测一体场景生成赛道的 `xosc` 场景生成任务。项目采用**经适配调整的数据驱动多智能体行为生成模型**，根据输入场景中的历史信息与 OpenDRIVE 路网信息，生成**背景车辆**行为并输出符合 OnSite 提交格式的 `*_output.xosc` 文件。**Ego 车辆不参与模型仿真**，仅保留输入场景中提供的历史轨迹。项目同时提供 GIF 可视化脚本，支持批量将 `xosc + xodr` 渲染为动画，便于检查背景车辆行为、驶出路网删除逻辑和整体场景效果。


<p align="center">
  <img src="asset/scenario_2a541f29_output.gif" width="32%" alt="OnSite generated scenario demo 1" />
  &nbsp;
  <img src="asset/scenario_2be310f8_output.gif" width="32%" alt="OnSite generated scenario demo 2" />
  &nbsp;
  <img src="asset/scenario_3011a285_output.gif" width="32%" alt="OnSite generated scenario demo 3" />
</p>

<p align="center">
  <img src="asset/scenario_3d95e43c_output.gif" width="32%" alt="OnSite generated scenario demo 4" />
  &nbsp;
  <img src="asset/scenario_4f89c34b_output.gif" width="32%" alt="OnSite generated scenario demo 5" />
  &nbsp;
  <img src="asset/scenario_5cb17c7e_output.gif" width="32%" alt="OnSite generated scenario demo 6" />
</p>

<p align="center"><em>图中 Ego 车辆未参与模型仿真，GIF 中仅回放输入场景提供的历史轨迹；背景车辆由模型生成并完成仿真。</em></p>

## 目录

* [1 环境配置](#jump1)
* [2 数据准备](#jump2)
* [3 文件说明](#jump3)
* [4 运行测试](#jump4)
* [5 可视化说明](#jump5)
* [6 致谢](#jump6)

## <span id="jump1">1 环境配置</span>

+ 创建并激活环境：

```bash
conda create -y -n onsite python=3.11.9
conda activate onsite
conda install -y -c conda-forge ffmpeg=4.3.2
pip install -r requirements.txt
pip install torch_geometric
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install --no-deps waymo-open-dataset-tf-2-12-0==1.6.4
```

> **备注：** 为了兼容性，您可以尝试不同的cuda版本，11.3、11.6、11.8已被确认是可以正常运行工作。

## <span id="jump2">2 数据准备</span>

每个测试场景目录需包含一个 OpenSCENARIO 文件和一个 OpenDRIVE 路网文件，命名格式如下：

```text
data/B/
└── scenario_xxx/
    ├── scenario_xxx_exam.xosc      # 输入场景，包含 Ego 与背景车历史信息
    └── scenario_xxx.xodr           # OpenDRIVE 路网文件
```

模型权重默认放置在 `ckpt/` 目录下。当前可用权重示例：

```text
ckpt/epoch=07-step=30440-val_loss=2.52.ckpt
```

## <span id="jump3">3 文件说明</span>

### 3.1 文件结构说明

```text
SMART-Onsite/
├── onsite_gen.py                   # 批量生成 *_output.xosc
├── visualize_xosc_gif.py           # 将 xosc + xodr 可视化为 GIF
├── inference.py                    # 模型推理与地图解析入口
├── requirements.txt                # Python 环境依赖
├── README.md                       # 英文说明文档
├── README_zh.md                    # 中文说明文档
├── asset/                          # README 使用的 logo 与示例 GIF
├── configs/validation/             # 模型配置文件
├── ckpt/                           # 模型权重目录
├── data/B/                         # B 榜输入测试场景
├── results/                        # 闭环评测汇总结果与逐场景得分
│   ├── summary_averages_100.csv    # 汇总平均得分
│   ├── per_scene_detailed_100.csv  # 逐场景详细得分
│   └── evaluation/                 # 闭环评测输出目录
├── smart/                          # 经适配调整的模型代码
└── utils/opendrive2discretenet/    # OpenDRIVE 路网解析工具
```

### 3.2 核心脚本说明

| 文件名 | 功能 |
|:---:|:---|
| `onsite_gen.py` | 批量读取 `*_exam.xosc`，调用适配后的模型生成背景车行为，并写入 `*_output.xosc` |
| `visualize_xosc_gif.py` | 批量或单文件渲染 `xosc + xodr` 为 GIF，支持图例、固定视角和驶出路网车辆删除显示 |
| `inference.py` | 模型推理、地图解析、场景数据构造等底层逻辑 |
| `requirements.txt` | Python 环境依赖 |

## <span id="jump4">4 运行测试</span>

### 4.1 批量生成 output xosc

运行以下命令，为测试目录中的所有场景批量生成 `*_output.xosc` 文件：

```bash
python onsite_gen.py \
  --test_dir data/B \
  --output_dir scene_sub \
  --ckpt_path ckpt/epoch=07-step=30440-val_loss=2.52.ckpt \
  --sampling_mode greedy \
  --seed 2026 \
  --road_exit_distance 5.0
```

主要参数说明：

| 参数 | 说明 |
|:---|:---|
| `--test_dir` | 输入场景根目录，每个子目录包含 `*_exam.xosc` 与 `.xodr` |
| `--output_dir` | 生成的 `*_output.xosc` 保存目录 |
| `--ckpt_path` | 适配模型的权重路径 |
| `--sampling_mode` | 推理采样策略，推荐使用 `greedy` 保持结果稳定 |
| `--road_exit_distance` | 背景车距离车道中心线超过该阈值时触发删除逻辑 |

#### 预期结果

下表为本基线在 **B 榜**（`450` 个场景）上的**预期闭环评测得分**，使用默认权重与 `greedy` 采样策略，对所有评测场景取平均。

| 指标 | 数值 |
|:---|---:|
| 赛道 | B |
| 场景数量 | 450 |
| BV Safety (20) | 18.178 |
| BV Comfort (10) | 6.393 |
| BV Test (30) | 7.963 |
| **BV Total (60)** | **32.533** |
| AV Safety (10) | 4.515 |
| AV Efficiency (10) | 3.838 |
| AV Comfort & Traffic Coordination (5 + 5) | 4.659 |
| AV Compliance (10) | 3.781 |
| **AV Total (40)** | **16.792** |
| **Total Score (100)** | **49.325** |

> **说明：** `BV` 表示背景车评测（60 分），`AV` 表示主车闭环评测（40 分）。场景生成阶段 Ego 车辆不参与模型仿真，仅保留输入场景中的历史轨迹。

### 4.2 单场景调试

如只想快速调试少量场景，可使用 `--limit`：

```bash
python onsite_gen.py \
  --test_dir data/B \
  --output_dir scene_sub \
  --ckpt_path ckpt/epoch=07-step=30440-val_loss=2.52.ckpt \
  --sampling_mode greedy \
  --seed 2026 \
  --limit 10
```

## <span id="jump5">5 可视化说明</span>

### 5.1 批量可视化

```bash
python visualize_xosc_gif.py \
  --scene_dir scene_sub \
  --xodr_root data/B \
  --output_dir scene_sub/gifs \
  --camera static \
  --view traj \
  --padding 4
```

### 5.2 单文件可视化

```bash
python visualize_xosc_gif.py \
  --xosc_path scene_sub/scenario_xxx_output.xosc \
  --xodr_path data/B/scenario_xxx/scenario_xxx.xodr \
  --output_gif scene_sub/gifs/scenario_xxx_output.gif \
  --camera static \
  --view traj \
  --padding 4
```

### 5.3 可视化参数说明

| 参数 | 说明 |
|:---:|:---|
| `--camera static` | 固定视角，推荐使用，可避免 GIF 画面跟随车辆晃动 |
| `--camera follow` | 跟随车辆视角，适合局部场景调试 |
| `--view traj` | 按车辆活动范围确定画布，画面更聚焦车辆 |
| `--view map` | 按完整路网范围确定画布，适合检查地图整体结构 |
| `--view both` | 同时考虑车辆活动范围和地图范围 |
| `--padding` | 画面边距，数值越小越聚焦车辆，常用范围为 `2` 到 `6` |

可视化默认使用 `8:6` 画布，并显示图例。脚本会读取 `DeleteEntityAction`，被删除的背景车辆会在对应时间后从 GIF 中消失。

## <span id="jump6">6 致谢</span>

衷心感谢国家自然科学基金委员会工程与材料科学部和中国汽车工程学会的支持以及[TOPS课题组](https://tops.tongji.edu.cn/index.htm)的集体努力与卓越贡献。感谢 [SMART](https://github.com/rainmaker22/SMART) 代码仓库提供了具有重要参考价值的代码，本仓库在其基础上进行了适配与扩展，对本工作产生了重要影响。
