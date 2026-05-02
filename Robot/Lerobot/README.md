# LeRobot 总览

LeRobot 不是某一个模型，也不是某一篇论文的方法，而是 Hugging Face 做的机器人学习框架。它把数据采集、数据格式、策略训练、仿真评估、真实机器人部署和模型发布放到同一套工程接口里。

如果只说一句话：LeRobot 负责把“机器人学习项目”整理成统一流程；ACT、Diffusion Policy、SmolVLA、pi0 这些才是具体 policy。

## 学什么

学习 LeRobot 不建议从“所有代码都看一遍”开始，而是抓住一条最小主线：

```text
robot/env observation
-> LeRobotDataset / observation schema
-> processor
-> policy
-> train / eval
-> action postprocess
-> robot/env step
```

也就是先看数据怎么进来，再看 policy 怎么吃数据，最后看动作怎么回到机器人或仿真环境。

## 框架分层

### 1. 数据层

核心问题：一条机器人示教数据在 LeRobot 里长什么样。

常见字段：

```text
observation.images.<camera_name>
observation.state
task
action
action_is_pad
episode_index
frame_index
timestamp
```

对于 VLA policy，还会出现语言相关字段：

```text
observation.language.tokens
observation.language.attention_mask
```

这一层最重要的是理解：LeRobot 会把不同机器人、不同相机、不同任务的数据统一成一个可以被 policy 使用的 schema。

#### 数据层代码示例

我把数据层单独拆成了几个官方风格的 Python 示例，放在：

```text
data_layer_examples/
```

建议按这个顺序读：

```text
01_load_and_inspect_dataset.py
02_dataloader_batch_shapes.py
03_record_dataset_official_skeleton.py
04_replay_episode_official_skeleton.py
05_custom_robot_interface_contract.py
```

其中前两个不需要真机，主要用来理解 LeRobotDataset 返回的字段和 batch shape。当前默认使用已经下载到本地的小型完整数据集：

```text
datasets/svla_so101_pickplace/
```

这个数据集来自 ModelScope `lerobot/svla_so101_pickplace`，包含 50 个 episode、11939 帧、两路视频、6 维状态和 6 维动作，适合作为数据层入门样本。

如果想看更标准的官方 LeRobot 数据集，可以使用已经下载好的：

```text
datasets/libero/
```

它对应 Hugging Face 官方 `lerobot/libero` 数据集，约 1.94 GB，包含 1693 个 episode、273465 帧、40 个任务、两路 256x256 视频、8 维状态和 7 维动作，更适合理解 VLA/多任务机器人数据如何组织。

```bash
python data_layer_examples/01_load_and_inspect_dataset.py --index 0
python data_layer_examples/02_dataloader_batch_shapes.py --batch-size 4
python data_layer_examples/04_replay_episode_official_skeleton.py --dry-run --max-steps 5

python data_layer_examples/01_load_and_inspect_dataset.py --repo-id lerobot/libero --root datasets/libero --index 0
python data_layer_examples/02_dataloader_batch_shapes.py --repo-id lerobot/libero --root datasets/libero --batch-size 4
```

后三个是官方文档风格的真机骨架，需要改端口、相机、机器人类型和 Hugging Face repo id：

- `03_record_dataset_official_skeleton.py`：官方 record 流程，重点看 `robot.observation_features`、`robot.action_features` 如何变成 dataset schema。
- `04_replay_episode_official_skeleton.py`：官方 replay 流程，重点看 `dataset.features["action"]["names"]` 如何把 action 向量还原成机器人命令字典。
- `05_custom_robot_interface_contract.py`：自定义硬件接口，重点看 `get_observation()`、`send_action()`、`observation_features`、`action_features` 四个核心 contract。

理解这几个文件之后，再看 ego 数据会更清楚：ego 论文本质上是在把 human ego video 补成类似 LeRobot 的 `observation -> action` episode，只是 action 可能来自 wrist pose、MANO、object 6DoF、retargeting 或 latent action。

### 2. Processor 层

Processor 是 LeRobot 很关键的一层，它负责把“原始观测”变成“模型真正需要的输入”。

可以分成两类：

```text
environment processor
policy processor
```

environment processor 处理环境差异，比如 LIBERO 的 nested robot state、相机方向、坐标系约定。

policy processor 处理模型需求，比如图像 resize/normalize、状态归一化、添加 batch 维度、移动到 GPU、语言 tokenization。

理解 processor 之后，很多“为什么这个 key 不见了 / 为什么 state 变成 8 维 / 为什么 image 被旋转了”的问题都会变清楚。

### 3. Policy 层

Policy 是具体学习算法或基础模型所在的位置。

可以粗略分成几类：

```text
ACT                 模仿学习 baseline，适合先理解 chunked action
Diffusion Policy    用扩散模型生成动作序列
SmolVLA             小型 VLA，图像 + 状态 + 语言 -> 连续动作块
pi0 / pi0.5         更大的 VLA / flow matching 路线
TD-MPC              强化学习 / model-based 控制路线
```

所以“学 LeRobot”一定会落到某个 policy 上。LeRobot 是工程框架，policy 是学习算法本体。

### 4. 训练与评估层

主要入口：

```text
lerobot-train
lerobot-eval
lerobot-record
```

`lerobot-train` 串起 dataset、processor、policy、optimizer 和 checkpoint。

`lerobot-eval` 串起 environment、processor、policy 和 rollout。

`lerobot-record` 面向真实机器人，负责采集数据或边运行 policy 边录制 episode。

### 5. Robot / Env 层

Robot 层处理真实硬件，例如 SO100/SO101、ALOHA、相机、leader-follower teleoperation。

Env 层处理仿真环境，例如 LIBERO、PushT 等。

对不接真机的学习来说，可以先走：

```text
LIBERO / PushT -> processor -> policy -> lerobot-eval
```

这样能先理解完整闭环，而不用被硬件连接、串口、相机驱动卡住。

## 多模态输入如何进入模型

以 SmolVLA 为例，输入不是全部变成离散 token id，而是都变成 transformer 能处理的序列 embedding。

### Language instruction

语言是真正的 tokenizer 路线：

```text
"pick up the cube"
-> tokenizer
-> token ids
-> attention mask
-> text embeddings
```

### Images

图像不是先转成文字 token，而是走视觉编码器：

```text
RGB images
-> resize / pad / normalize
-> vision encoder
-> patch embeddings
```

这些 patch embeddings 可以理解成视觉 token，但它们是连续向量，不是离散词表 id。

### Robot state

机器人状态通常是连续向量：

```text
joint / eef / gripper state
-> observation.state
-> normalize
-> pad 到统一维度
-> linear projection
-> state embedding
```

state embedding 也可以作为 token 放进 transformer 上下文。

### Action

SmolVLA 输出的是连续动作块，不是 OpenVLA 那种离散 action token：

```text
noisy action chunk + time
-> action expert
-> continuous action chunk
```

这也是 SmolVLA 和 OpenVLA 很重要的区别：OpenVLA 把动作离散化为 token，SmolVLA / pi0 更偏连续动作建模。

## 推荐源码阅读顺序

第一轮只看主线，不追细枝末节：

```text
src/lerobot/datasets/
src/lerobot/processor/
src/lerobot/policies/act/
src/lerobot/policies/smolvla/
src/lerobot/scripts/lerobot_train.py
src/lerobot/scripts/lerobot_eval.py
src/lerobot/envs/
src/lerobot/robots/
```

建议顺序：

1. 先看 dataset 的字段和 stats。
2. 再看 processor 怎样改 batch。
3. 用 ACT 理解最小 imitation learning policy。
4. 用 SmolVLA 理解 VLA policy 怎样吃 image/state/language。
5. 最后看 train/eval 脚本如何把这些模块接起来。

## 和当前笔记的关系

- `Diffusion_Policy/`：模仿学习 baseline，适合理解动作序列建模。
- `VLM/SmolVLA/`：LeRobot 里的 VLA policy 学习入口。
- `VLM/OpenVLA/`：离散 action token 路线，可以和 SmolVLA 对比。
- `VLM/pi0/`：连续 action + flow matching 路线，可以和 SmolVLA 对比。
- `ROS2/`：后续接真实机器人系统时需要的通信基础。
- `Gymnasium_Robotics/`、`EI_Mujoco/`：仿真和强化学习环境基础。

## 最小实践路线

1. 读 LeRobot 数据格式，知道一条 episode 有哪些字段。
2. 跑一个小 dataset 的加载和可视化。
3. 跑 ACT 或 Diffusion Policy 的最小训练命令。
4. 跑 SmolVLA 的单 batch 推理，检查 image/state/language/action 的 shape。
5. 在 LIBERO 或 PushT 上跑少量 episode 的 eval。
6. 再考虑真实 SO100/SO101 或自己的机器人数据采集。

## 关键判断

LeRobot 本身不是终点，它是入口。

真正要学透，需要选一个 policy 作为主线。当前最合适的组合是：

```text
LeRobot 框架总览
-> ACT / Diffusion Policy 理解模仿学习基础
-> SmolVLA 理解 VLA 工程闭环
-> pi0 / OpenVLA 做方法对比
```

## 资料入口

- LeRobot GitHub：<https://github.com/huggingface/lerobot>
- LeRobot 文档：<https://huggingface.co/docs/lerobot>
- SmolVLA 文档：<https://huggingface.co/docs/lerobot/smolvla>
- Environment processor 文档：<https://huggingface.co/docs/lerobot/env_processor>
