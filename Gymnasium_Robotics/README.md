# Gymnasium Robotics 任务路线

这个文件夹对应任务二从通用 RL 环境切到机器人控制环境的过渡阶段。

选择 `Fetch` 系列的原因是：它已经非常接近机械臂操作任务，同时动作空间又足够直观。末端执行器通过笛卡尔位移控制，逆运动学由 MuJoCo 在环境内部处理，非常适合作为从通用 RL 迈向机械臂抓取的桥梁。

## 主线任务顺序

1. `FetchReach-v3`
   - 目标：先学会稳定控制末端到达目标点
2. `FetchPush-v3`
   - 目标：从纯位姿控制进入物体交互
3. `FetchPickAndPlace-v3`
   - 目标：完成完整的抓取与移动任务

## 推荐算法

- 如果先做 dense reward 版本：`PPO` 或 `SAC`
- 如果做 sparse reward 版本：优先考虑 `HER + SAC` 或 `HER + TD3`

## 工程备注

- `gymnasium-robotics` 需要 MuJoCo
- 官方当前测试与支持重点在 Linux 和 macOS
- Windows 可以尝试，但更推荐 `WSL2 / Ubuntu / 远程 Linux` 环境

## 本阶段完成标准

- 能跑通至少一个 `Fetch` 环境训练
- 能理解 goal-conditioned observation 的结构
- 能说清 dense reward 与 sparse reward 的区别
- 能总结一个最基础的机器人 RL 训练流程

## 参考链接

- Fetch index: <https://robotics.farama.org/envs/fetch/index.html>
- FetchReach: <https://robotics.farama.org/envs/fetch/reach/>
- Installation: <https://robotics.farama.org/content/installation/>
- Stable-Baselines3 RL Tips: <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>
