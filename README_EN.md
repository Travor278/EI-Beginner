# Embodied Intelligence — Personal Learning Notes

[中文版本](./README.md) | [English Version](./README_EN.md)

This repository contains my personal notes and practice code for learning embodied intelligence and humanoid robotics, covering simulation setup, reinforcement learning, imitation learning, and more.

## Repository Contents

| File / Folder | Description |
| --- | --- |
| `EI_Mujoco/hello_mujoco.py` | MuJoCo starter exercise: build a simple scene, load a physics model, run a simulation loop |
| `linux指令.txt` | Linux command cheatsheet for ML/robotics workflows — file ops, process management, GPU monitoring, training logs |

## Prerequisites

- Comfortable with ChatGPT/DeepSeek and Google
- Comfortable with Linux
- Comfortable with Git and GitHub
  - <https://learngitbranching.js.org/>

## Task 1: Traditional Kinematics-based Robotic Arm Grasping

![Task 1](./assets/task1.png)

Learn core classical robotics concepts — coordinate transforms, forward/inverse kinematics, dynamics, and control theory — then implement robotic arm object grasping in PyBullet/MuJoCo simulation using traditional motion control.

References:

- Introduction to Robotics: Mechanics and Control — Stanford
- Robotic Manipulation — MIT
- PyBullet: <https://github.com/bulletphysics/bullet3>
- MuJoCo: <https://mujoco.org/>
- Challenge game: <https://rcfs.ch/>

## Task 2: Reinforcement Learning-based Robotic Arm Grasping

![Task 2](./assets/task2.png)

1. Learn RL fundamentals and train agents on several tasks in OpenAI Gym;
2. Train a grasping policy in PyBullet/MuJoCo and experience the Sim2Real transfer process.

References:

- Introduction to Reinforcement Learning, 2nd Ed. & David Silver's UCL Course
- UCB CS285 Deep Reinforcement Learning
- OpenAI Gym: <https://gymnasium.farama.org/index.html>

## Task 3: Imitation Learning-based Robotic Arm Grasping

![Task 3](./assets/task3.png)

1. Reproduce the classic imitation learning baseline Diffusion Policy
   - <https://diffusion-policy.cs.columbia.edu>
2. Learn the HuggingFace robotics framework LeRobot
   - <https://github.com/huggingface/lerobot>

## Task 4: VLA Large Model-based Robotic Arm Grasping

![Task 4](./assets/task4.png)

Study existing VLA (Vision-Language-Action) models such as OpenVLA, Pi, and GR00T, and explore training specialized VLA models with open robotic datasets.

References:

- OpenVLA: <https://github.com/openvla/openvla>
- Pi: <https://github.com/Physical-Intelligence/openpi>
- GR00T: <https://github.com/NVIDIA/Isaac-GR00T>
- Open-X Embodiment: <https://robotics-transformer-x.github.io/>
- Large Models: <https://stanford-cs336.github.io/spring2025/>

## Task 5: LLM/VLM-based Task Planning

![Task 5a](./assets/task5_1.png)

- Desktop-level task planning
  - Reference: "Code as Policies" (ICRA 2023 Outstanding Paper) — <https://code-as-policies.github.io/>
  - Prompt existing LLMs/VLMs to complete the task;
  - Fine-tune existing LLMs/VLMs to complete the task.

![Task 5b](./assets/task5_2.png)

- Scene-level task planning
  - Set up a simulation environment and run a baseline;
  - Design ICL or CoT methods to improve embodied planning performance.

Optional simulation environments / benchmarks:

- EAI: <https://github.com/embodied-agent-interface/embodied-agent-interface>
- EmbodiedBench: <https://github.com/EmbodiedBench/EmbodiedBench>

Reference papers:

- [Embodied Agent Interface](https://arxiv.org/pdf/2410.07166)
- [VisualAgentBench](https://arxiv.org/pdf/2408.06327)
- [LLM-Planner](https://arxiv.org/abs/2212.04088)
- [ReAct](https://arxiv.org/pdf/2210.03629)

## Task 6: Reinforcement Learning-based Humanoid Robot Locomotion

![Task 6](./assets/task6.png)

Reproduce the humanoid locomotion control method from [OmniH2O](https://omni.human2humanoid.com/) and learn the simulation training + Sim2Real pipeline.

References:

- Unitree Robotics GitHub: <https://github.com/unitreerobotics>
- HOVER: <https://github.com/NVlabs/HOVER>
- Underactuated Robotics — MIT

## Frontier Research

- How to do research
  - [An Opinionated Guide to ML Research](http://joschu.net/blog/opinionated-guide-ml-research.html)
  - [GAMES003: Basic Literacy in Graphics & Vision Research](https://pengsida.net/games003/)
- Conferences & Journals
  - Robotics: Science Robotics, RSS, CoRL, ICRA, IROS, RLC
  - Machine Learning: ICLR, NeurIPS, ICML
  - Computer Vision: CVPR, ICCV, ECCV
  - NLP: ACL, EMNLP, COLM
- Online Seminars
  - [CMU Robotics Institute Seminar](https://www.youtube.com/@cmurobotics)
  - [MIT Robotics Seminar](https://www.youtube.com/@MITRoboticsSeminar)
  - [MIT Embodied Intelligence Seminar](https://www.youtube.com/@mitembodiedintelligence8675)
  - [Stanford Seminar](https://www.youtube.com/@stanfordonline)
