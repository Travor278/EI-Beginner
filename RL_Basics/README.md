# 强化学习基础学习路径

这个文件夹对应任务二的第一阶段。

目标不是一次性把整本强化学习教材全部看完，而是建立一条够用、能立刻落到环境训练里的学习路线。建议按 "理解概念 -> 做小实验 -> 写总结" 的节奏推进。

## 推荐学习顺序

1. 理解 `agent / environment / state / action / reward / return`
2. 理解 `policy`、`value function`、`Bellman equation`
3. 学习 `multi-armed bandit` 与探索-利用权衡
4. 学习 `Dynamic Programming`、`Monte Carlo`、`Temporal Difference`
5. 学习 `SARSA` 和 `Q-learning`
6. 理解函数逼近、`DQN` 的基本动机
7. 理解 `Policy Gradient`、`Actor-Critic`、`PPO / SAC` 的直觉

## 推荐资料

- Sutton & Barto, *Reinforcement Learning: An Introduction (2nd Edition)*
  - 先看前半部分的表格型方法，再进入函数逼近与策略梯度
- David Silver RL Course
  - 用来建立直觉，适合理解价值函数、Bellman 方程和策略改进
- UCB CS285
  - 在已经有基础概念后再看，更适合把经典 RL 过渡到现代深度强化学习
- Gymnasium Migration Guide
  - 用来适应新版 API，避免把旧版 `gym` 教程直接照搬

## 本阶段完成标准

- 能用自己的话解释 `MDP`、`return`、`value`、`policy`
- 能说明 `MC`、`TD`、`SARSA`、`Q-learning` 的区别
- 能手写一个表格型 `Q-learning` baseline
- 能正确处理 Gymnasium 的 `reset()` 和 `step()` 返回值

## 对应到后续任务

- 学完表格型方法后，先做 `FrozenLake-v1`
- 学完离散动作深度 RL 后，进入 `CartPole-v1`
- 学完连续动作与 actor-critic 后，进入 `Pendulum-v1`
- 然后再进入 `FetchReach -> FetchPush -> FetchPickAndPlace`

## 参考链接

- Sutton & Barto (MIT Press): <https://mitpress.mit.edu/9780262352703/reinforcement-learning/>
- David Silver RL Course: <https://www.youtube.com/watch?v=2pWv7GOvuf0>
- CS285: <https://www2.eecs.berkeley.edu/Courses/CS285/>
- Gymnasium Migration Guide: <https://gymnasium.farama.org/main/introduction/migration_guide/>
