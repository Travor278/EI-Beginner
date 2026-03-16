import mujoco
import mujoco.viewer
import time

# 1. 定义一个简单的模型 (XML 格式)
# 包含：一个光源、一个无限大的平面（地面）、一个带有自由关节的红色方块
xml_string = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .9 .8 1"/>
    
    <body name="box" pos="0 0 1">
      <joint type="free"/> <geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 2. 将 XML 字符串加载为 MuJoCo 的模型 (Model) 和数据 (Data)
# Model 包含静态信息（比如物体的形状、质量、关节结构）
model = mujoco.MjModel.from_xml_string(xml_string)
# Data 包含动态信息（比如当前的位置、速度、受力情况）
data = mujoco.MjData(model)

# 3. 启动被动模式的可视化器 (Viewer)
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 记录仿真开始时间
    start = time.time()
    
    # 只要窗口没被关闭，就一直循环
    while viewer.is_running() and time.time() - start < 10:
        step_start = time.time()
        
        # 核心：让物理引擎计算下一帧的状态
        mujoco.mj_step(model, data)
        
        # 将计算出的新状态同步到画面上
        viewer.sync()
        
        # 控制仿真速度，使其大致符合真实时间
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)