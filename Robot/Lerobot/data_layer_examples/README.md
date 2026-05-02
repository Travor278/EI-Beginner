# LeRobot 数据层示例怎么跑

这个文件夹用来理解 LeRobot 的数据层：一条机器人示教数据如何从 `observation`、`action`、`task` 变成 policy 可以吃的 batch。

建议先跑 01 和 02，它们不需要真机。03 到 05 是官方风格的真机/接口骨架，主要用来读代码结构。

## 0. 环境

请用 `py312`，不要用 `py313`。`py313` 里的 `torchvision 0.26` 没有 `torchvision.io.VideoReader`，读取视频时会报：

```text
AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
```

PowerShell:

```powershell
conda activate py312
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

如果你想完全避免路径歧义，也可以直接用：

```powershell
D:\Dev\conda-envs\py312\python.exe
```

## 1. 数据集

默认数据集是本地小样本：

```text
..\datasets\svla_so101_pickplace
```

它包含 50 个 episode、两路视频、6 维 `observation.state` 和 6 维 `action`，适合第一次看数据格式。

更官方、更适合理解 VLA/多任务数据层的是：

```text
..\datasets\libero
```

它对应 `lerobot/libero`，包含 1693 个 episode、40 个任务、两路 256x256 视频、8 维 state 和 7 维 action。

## 2. 推荐顺序

### 01: 看单帧 sample

默认跑小样本：

```powershell
python .\01_load_and_inspect_dataset.py --index 0
```

跑官方 `libero`：

```powershell
python .\01_load_and_inspect_dataset.py --repo-id lerobot/libero --root ..\datasets\libero --index 0
```

看时间窗口输入，也就是一个图像 key 返回多帧历史：

```powershell
python .\01_load_and_inspect_dataset.py --index 10 --with-history
```

你应该重点看这些字段：

```text
observation.images.*
observation.state
action
task
timestamp
episode_index
frame_index
```

### 02: 看 DataLoader batch

默认跑小样本：

```powershell
python .\02_dataloader_batch_shapes.py --batch-size 4
```

跑官方 `libero`：

```powershell
python .\02_dataloader_batch_shapes.py --repo-id lerobot/libero --root ..\datasets\libero --batch-size 4
```

你会看到类似：

```text
observation.images.image:  [B, C, H, W]
observation.state:         [B, state_dim]
action:                    [B, action_dim]
task:                      list[len=B]
```

这一步最重要：policy 训练时看到的不是单帧，而是 batch。

### 03: 官方 record 骨架

默认是 dry run，只打印由机器人接口推出来的 dataset schema，不会连接真机：

```powershell
python .\03_record_dataset_official_skeleton.py
```

确认串口、相机、repo_id、task 都改好以后，再加 `--record` 真正录制：

```powershell
python .\03_record_dataset_official_skeleton.py --record
```

重点看：

```text
robot.observation_features
robot.action_features
aggregate_pipeline_dataset_features(...)
LeRobotDataset.create(...)
record_loop(...)
```

它说明真实机器人的观测/动作接口如何经过 processor feature transform 变成 LeRobot dataset schema。

### 04: replay 动作

默认 dry run，不连真机，只打印 action 字典：

```powershell
python .\04_replay_episode_official_skeleton.py --dry-run --max-steps 5
```

跑 `libero` 的 dry run：

```powershell
python .\04_replay_episode_official_skeleton.py --repo-id lerobot/libero --root ..\datasets\libero --dry-run --max-steps 5
```

重点看：

```text
dataset.features["action"]["names"]
action vector -> action dict
```

这能帮助理解数据里的连续 action 如何还原成机器人命令。

### 05: 自定义 Robot 接口

这个文件主要用来读，不需要直接跑。重点看一个自定义机器人最少要提供什么：

```text
observation_features
action_features
get_observation()
send_action()
```

理解它之后，再看 ego 数据会更自然：ego 数据也要被整理成类似的 `observation -> action` episode，只是 action 可能来自 wrist pose、MANO、object pose、retargeting 或 latent action。

## 3. 常见问题

### torchcodec warning

看到这个通常不严重：

```text
'torchcodec' is not available in your platform, falling back to 'pyav'
```

Windows 上很常见。只要 `pyav`、`ffmpeg`、`torchvision` 当前组合能解码视频，01/02 能正常输出图像 tensor，就可以继续学。

### 跑着跑着去下载 HF 大数据

如果本地 `--root` 写错，脚本可能回退到 Hub 下载。优先使用这些本地路径：

```text
..\datasets\svla_so101_pickplace
..\datasets\libero
```

### PowerShell 显示 py312 但实际用了 py313

如果你这样跑：

```powershell
& D:\Dev\conda-envs\py313\python.exe .\01_load_and_inspect_dataset.py
```

即使提示符显示 `(py312)`，实际仍然是 `py313`。请改成：

```powershell
python .\01_load_and_inspect_dataset.py
```

或显式使用：

```powershell
D:\Dev\conda-envs\py312\python.exe .\01_load_and_inspect_dataset.py
```
