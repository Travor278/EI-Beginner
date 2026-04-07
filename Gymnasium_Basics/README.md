# Gymnasium 入门任务路线

这个文件夹对应任务二里 "先在通用环境中训练并测试" 的部分。

这里不是随机挑环境，而是按难度、动作空间类型和与机械臂任务的相关性排好了一条主线。主线完成后，再进入机器人环境会顺很多。

## 主线任务顺序

1. `FrozenLake-v1`
   - 目标：理解表格型 RL、稀疏奖励、探索问题
   - 推荐算法：`Q-learning`
2. `CartPole-v1`
   - 目标：完成第一个离散动作控制任务
   - 推荐算法：`DQN`，也可以用 `PPO` 做对照
3. `Pendulum-v1`
   - 目标：进入连续动作控制
   - 推荐算法：`PPO` 或 `SAC`
4. `LunarLander-v3`
   - 目标：作为综合练习，检验离散或连续控制能力
   - 推荐算法：离散版用 `DQN / PPO`，连续版用 `PPO / SAC`

## 建议推进方式

- 如果时间紧，主线做到 `Pendulum-v1` 就可以准备切机器人环境
- 如果想把 Gymnasium 基础打得更扎实，再补 `LunarLander-v3`
- 每个环境都至少做一次训练曲线、一次测试 rollout、一次失败案例分析

## 每个任务建议保留的内容

- 环境说明与状态/动作空间笔记
- 训练脚本
- 超参数记录
- 训练曲线截图
- 一段测试结果总结

## Gymnasium API 备注

- 使用 `import gymnasium as gym`
- `obs, info = env.reset()`
- `obs, reward, terminated, truncated, info = env.step(action)`
- 训练循环结束条件应写成 `terminated or truncated`

## 参考链接

- Classic Control: <https://gymnasium.farama.org/main/environments/classic_control/>
- CartPole: <https://gymnasium.farama.org/environments/classic_control/cart_pole/>
- Stable-Baselines3 RL Tips: <https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html>
- Stable-Baselines3 Algorithms: <https://stable-baselines3.readthedocs.io/en/master/guide/algos.html>
