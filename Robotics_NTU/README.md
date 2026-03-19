# 台大《机器人学》课堂笔记

> 国立台湾大学 林沛群教授《机器人学》自学笔记整理

## 目录

| 章节 | 笔记 | 可视化 |
|------|------|--------|
| 第一章：位置描述与旋转矩阵基础 | [Chapter01_Motion_and_Rotation.md](Chapter01_Motion_and_Rotation.md) | [.py](Chapter01_visualization.py) |
| 第二章：旋转的多种表示方法 | [Chapter02_Rotation_Representation.md](Chapter02_Rotation_Representation.md) | [.ipynb](Chapter02_visualization.ipynb) |
| 第三章：机械手臂顺运动学 | [Chapter03_Forward_Kinematics.md](Chapter03_Forward_Kinematics.md) | [.ipynb](Chapter03_visualization.ipynb) |
| 第四章：机械手臂逆运动学 | [Chapter04_Inverse_Kinematics.md](Chapter04_Inverse_Kinematics.md) | [.ipynb](Chapter04_visualization.ipynb) |

## 内容概览

### 第一章
- 坐标系与位置描述
- 平移运动
- 旋转运动与旋转矩阵 $R \in SO(3)$
- $R_x$、$R_y$、$R_z$ 三个基本旋转矩阵
- 旋转的组合与不可交换性
- 自由度分析

### 第二章
- 旋转矩阵的三种角色（描述、坐标变换、算子）
- **Fixed Angles（固定角/RPY）**
- **Euler Angles（欧拉角）**
- **★ 左乘 vs 右乘的物理意义**
- Mapping（坐标系间映射）
- 齐次变换矩阵 $T \in SE(3)$
- Gimbal Lock（万向锁）

### 第三章
- 串联机械手臂的连杆与关节结构
- DH 四参数：$a_i,\alpha_i,d_i,\theta_i$
- 单节连杆的齐次变换
- 2R 平面机械臂顺运动学
- Joint Space / Cartesian Space 的关系

### 第四章
- 逆运动学的基本问题定义
- 2R 平面机械臂解析逆解
- 多解、无解与工作空间边界
- 奇异位形的几何直觉
- Pieper's Solution 与球腕结构
- Pick-and-Place 任务拆解

## 运行可视化

```bash
# 安装依赖
pip install numpy matplotlib ipywidgets

# 运行 Python 脚本
python Chapter01_visualization.py

# 运行 Jupyter Notebook（交互版 + 批量导出）
jupyter notebook Chapter02_visualization.ipynb
jupyter notebook Chapter03_visualization.ipynb
jupyter notebook Chapter04_visualization.ipynb
```

在各章节 Notebook 的导出单元中运行对应函数，可将图片导出到 `Robotics_NTU/images/`。
