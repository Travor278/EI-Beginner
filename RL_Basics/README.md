# 强化学习基础学习路径

这个文件夹对应任务二的第一阶段。

目标不是一次性把整本强化学习教材全部看完，而是建立一条够用、能立刻落到环境训练里的学习路线。建议按 "读一段文字资料 -> 做一个小实验 -> 写一段总结" 的节奏推进。

如果觉得纯视频太慢，这一阶段完全可以以**网页/讲义/笔记**为主，只把视频当补充。

## 推荐学习顺序

1. 理解 `agent / environment / state / action / reward / return`
2. 理解 `policy`、`value function`、`Bellman equation`
3. 学习 `multi-armed bandit` 与探索-利用权衡
4. 学习 `Dynamic Programming`、`Monte Carlo`、`Temporal Difference`
5. 学习 `SARSA` 和 `Q-learning`
6. 理解函数逼近、`DQN` 的基本动机
7. 理解 `Policy Gradient`、`Actor-Critic`、`PPO / SAC` 的直觉

## 推荐读法

优先顺序建议是：

1. 先读 `Easy RL（蘑菇书）`
   - 它最像系统化网页笔记，读起来比纯视频轻很多
2. 遇到概念不稳的地方，再翻 `Sutton & Barto`
   - 这本是理论主教材，用来补定义、公式和细节
3. 如果某个点还是没感觉，再看 `David Silver` 对应一讲
   - 只补需要的那一讲，不必从头把整套视频刷完

## 推荐资料与定位

- `Easy RL（蘑菇书）`
  - 最推荐的中文网页/笔记式资料，适合入门表格型方法
  - 适合先建立整体框架
- `Sutton & Barto, Reinforcement Learning: An Introduction (2nd Edition)`
  - 理论主教材，适合查概念、公式和算法细节
  - 这一阶段重点看前半本
- `Stanford CS234 Modules`
  - 结构非常清楚，适合作为英文版课程讲义主线
  - 如果想读更严谨的课程文字说明，可以用它
- `David Silver RL Course`
  - 更适合补直觉，不一定要整套看完
- `OpenAI Spinning Up`
  - 更偏深度强化学习，放到表格型方法之后再读
- `Gymnasium Migration Guide`
  - 用来适应新版 API，避免把旧版 `gym` 教程直接照搬

## 表格型方法先读哪些

如果现在就准备开始，建议先读这几块：

1. `Easy RL`
   - 先读强化学习基本概念
   - 再读 `MDP / Bellman Equation`
   - 然后读 `Dynamic Programming / Monte Carlo / TD / Q-learning / SARSA`
2. `Sutton & Barto`
   - Chapter 1: Introduction
   - Chapter 2: Multi-armed Bandits
   - Chapter 3: Finite Markov Decision Processes
   - Chapter 4: Dynamic Programming
   - Chapter 5: Monte Carlo Methods
   - Chapter 6: Temporal-Difference Learning
3. `Stanford CS234 Modules`
   - `Tabular MDP planning`
   - `Tabular RL policy evaluation`
   - `Model-free control`

## 最小执行路线

如果你不想在资料里犹豫太久，可以直接这么开始：

1. 先读 `Easy RL` 里关于 `MDP / Bellman / Q-learning` 的内容
2. 同时翻 `Sutton & Barto` 的 Chapter 3-6
3. 立刻开始手写 `FrozenLake-v1` 的 `Q-learning`
4. 学完后自己写一页总结：
   - `MC` 和 `TD` 有什么区别
   - `SARSA` 和 `Q-learning` 有什么区别
   - 稀疏奖励为什么难

## 不建议的学法

- 不建议先把所有视频刷完再开始动手
- 不建议上来就学 `PPO / SAC` 而跳过表格型方法
- 不建议只看博客碎片而没有一条主线

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

- Easy RL（GitHub）: <https://github.com/datawhalechina/easy-rl>
- Sutton & Barto (MIT Press): <https://mitpress.mit.edu/9780262352703/reinforcement-learning/>
- Sutton & Barto PDF: <http://incompleteideas.net/book/the-book-2nd.html>
- Stanford CS234: <https://web.stanford.edu/class/cs234/>
- Stanford CS234 Modules: <https://web.stanford.edu/class/cs234/modules.html>
- David Silver RL Course: <https://www.davidsilver.uk/teaching/>
- CS285: <https://www2.eecs.berkeley.edu/Courses/CS285/>
- OpenAI Spinning Up: <https://spinningup.openai.com/>
- Gymnasium Migration Guide: <https://gymnasium.farama.org/main/introduction/migration_guide/>
