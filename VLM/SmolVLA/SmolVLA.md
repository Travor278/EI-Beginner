### SmolVLA——把 VLA 从“大模型展示”拉回到消费级硬件和社区数据

论文：
[SmolVLA - A Vision-Language-Action Model for Affordable and Efficient Robotics.pdf](./SmolVLA%20-%20A%20Vision-Language-Action%20Model%20for%20Affordable%20and%20Efficient%20Robotics.pdf)

官方资源：
- 论文：<https://arxiv.org/abs/2506.01844>
- Hugging Face paper page：<https://huggingface.co/papers/2506.01844>
- 官方博客：<https://huggingface.co/blog/smolvla>
- 基础模型：<https://huggingface.co/lerobot/smolvla_base>
- LeRobot 文档：<https://huggingface.co/docs/lerobot/smolvla>
- 官方代码：<https://github.com/huggingface/lerobot>

#### 解决什么问题

1.现有 VLA 太大，训练和部署门槛高
OpenVLA、pi0、GR00T 这类路线已经证明 VLA 有价值，但很多模型参数量、训练成本和部署成本都偏高。SmolVLA 的核心立场是：机器人基础模型不一定必须从巨大模型开始，也可以走小模型、高质量社区数据、可复现实验的路线。

2.机器人数据不只是规模问题，还是开放性和格式问题
SmolVLA 强调 LeRobot 社区数据的价值：大量用户用低成本机器人采集数据，再用统一格式沉淀下来。这里的关键不是“某个实验室有私有大数据”，而是让普通研究者也能复用、检查、继续贡献数据。

3.动作预测延迟会直接影响机器人表现
VLA 不只是离线预测准确率问题。真实控制里，模型推理时机器人如果停住等结果，就会造成明显卡顿。SmolVLA 因此把异步推理作为系统贡献：机器人执行当前 action chunk 时，服务端提前预测下一段动作。

#### 核心方法

1.使用小型 VLM 作为视觉语言主干
SmolVLA 采用 SmolVLM 系列作为基础，把多视角图像、机器人状态和语言指令编码成上下文特征。

2.用 action expert 生成连续动作块
它不像 OpenVLA 那样把动作离散成 token 再自回归生成，而是输出连续动作，并使用 flow matching 训练动作专家。这一点更接近 pi0 的连续动作建模路线。

3.预训练数据来自 LeRobot 社区数据
官方博客强调，SmolVLA 使用兼容许可证的公开 LeRobot 社区数据进行预训练，并对任务标注、相机视角等做了清洗和标准化。

4.异步推理系统
SmolVLA 的系统侧重点是把动作预测和动作执行解耦：当前 chunk 还在执行时，下一轮推理已经开始；不同 chunk 的重叠部分通过聚合函数融合，减少等待推理造成的 idle frames。

#### 核心贡献

1.提出 450M 级别的开源 VLA
官方模型卡称 `lerobot/smolvla_base` 是一个紧凑、高效的 VLA，输入多视角图像、机器人状态和语言指令，输出连续动作，目标是作为基础模型再微调到具体任务。

2.证明社区数据预训练是有价值的
官方博客报告，在 SO100 任务上，不做 LeRobot 社区数据预训练时成功率为 51.7%，加入社区数据预训练后到 78.3%，提升 26.6 个百分点。

3.把 VLA 工程闭环做得更完整
论文、模型、数据、训练脚本、推理脚本、异步推理文档都放在 LeRobot 生态里，这比只发一个模型更适合学习和复现。

4.提供面向仿真的复现实入口
LeRobot 文档已经支持 LIBERO benchmark，能用 `lerobot-train` 和 `lerobot-eval` 跑 SmolVLA 训练/评估，不接真机也能做部分实验。

#### 和 OpenVLA / pi0 的关系

1.相比 OpenVLA，SmolVLA 更偏连续动作和低成本部署
OpenVLA 的重点是把 VLM 变成动作 token 生成器；SmolVLA 则保留 VLM 语义能力，同时用连续 action expert 预测动作块，减少离散化动作的接口损失。

2.相比 pi0，SmolVLA 更小、更社区化
pi0 强调大规模机器人数据、后训练和 flow matching 动作专家；SmolVLA 借鉴了连续动作块和 flow matching 方向，但把模型规模压到 450M，并把 LeRobot 社区数据作为核心叙事。

3.SmolVLA 最值得学的是工程路径
这篇论文的亮点不只是模型结构，而是 LeRobot 格式、数据清洗、训练脚本、策略接口、异步推理和仿真 benchmark 这些东西如何拼成一条能跑的路线。
