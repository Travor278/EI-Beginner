# FetchReach-v3

这是从通用控制转向机械臂控制的第一站。

## 目标

- 学会控制末端执行器到达目标位置
- 熟悉 goal-conditioned observation 和机器人环境的训练接口

## 为什么先做它

- 没有抓取和接触过程，问题更干净
- 任务本质接近 `reach-only`，非常适合作为机器人 RL 热身

## 完成标准

- 能稳定完成到达任务
- 能解释环境中的 `observation`、`achieved_goal`、`desired_goal`
