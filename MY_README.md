工作站
用户名：dell
密码：tsinghua

向日葵
识别码：556 656 698
长期验证码：h38qto

toDesk
设备代码：669 971 533
密码：641pqi7g

文件结构
├── assets
│   ├── 场地与机器人模型
├── cfg
│   ├── 环境与学习配置
├── checkpoints
│   ├── 模型保存
├── envs
│   ├── 环境Wrapper
├── models
│   ├── 接入的算法
├── robots
│   └── 场地搭建与机器人生成
├── runs
│   └── rlgames模型保存（可不管）
├── scripts
│   ├── 启动文件
├── tasks
│   ├── RL任务定义，可按需修改state,action,reward等
└── utils
    ├── 工具


读取镜像
docker load -i pursuit-sim.tar

创建容器（不想退出自动删除则去掉--rm）
docker run --name pursuit-sim-container -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY -e "PRIVACY_CONSENT=Y" -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw -v ~/docker/isaac-sim/documents:/root/Documents:rw -v ~/docker/isaac-sim/PursuitSim3D:/workspace/omniisaacgymenvs/PursuitSim3D:rw pursuit-sim

启动训练
/isaac-sim/python.sh scripts/train.py

启动测试
/isaac-sim/python.sh scripts/test.py test=True

打开pycharm
/workspace/omniisaacgymenvs/pycharm/bin/pycharm.sh

obs: 61维
[x1,y1,z1,r1,p1,yaw1,lidar_depth1,x2,y2,z2,r2,p2,yaw2,lidar_depth2,x3,y3,z3,r3,p3,yaw3,lidar_depth3,x4,y4,z4,r4,p4,yaw4,t]
其中lidar_depth包含12方位的数据
action: 6维
[x1,y1,x2,y2,x3,y3]

启动issac  

/isaac-sim/isaac-sim.sh --allow-root

安装库指令：
/isaac-sim/python.sh -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple

tensorboard指令:
/isaac-sim/python.sh -m tensorboard --logdir=logs/

本项目安装依赖：

/isaac-sim/python.sh -m pip install "stable-baselines3==2.4.0" "sb3-contrib==2.4.0" "shimmy==0.2.1" “openpyxl”

apt-get install -y tmux tree

启动容器后再开一个窗口命令行进入它：

docker ps -a

docker start pursuit-sim-container

docker exec -it pursuit-sim-container bash

删除容器

docker stop pursuit-sim-container

docker rm pursuit-sim-container

重启：
sudo reboot

sudo systemctl restart runsunloginclient



---

启动train前需要修改的路径：

1. train文件中诸如PPO.main的main函数的checkpoints存储位置
2. evaluation_for_training输入名字
3. 要load进来的model的名字
4. evaluation最后



选择的文件版本：
robogame_task.py、vec_env_rlgames2.py（新版gymnasium）、rl_task.py



各算法选择的checkpoints：

td3v8、SACv2、PPOv2、NPGv2，其他没有多版本

选择的地图：

map_simple（注：论文的高台和斜坡封闭，也即不可上去）

---

用“锁”把驱动版本锁死

**1. 一键锁定所有 NVIDIA 相关包** 复制并运行下面这行命令（这行命令会查找所有已安装的 nvidia 包并将其标记为“保留/hold”）：

```
dpkg -l | grep nvidia | awk '{print $2}' | xargs sudo apt-mark hold
```

**2. 验证是否锁成功** 运行：

```
apt-mark showhold
```

如果你看到一大堆 `libnvidia-xxx`, `nvidia-driver-xxx` 的输出，说明成功了。

如果你将来想主动升级驱动，需要先解锁：

```
# 解锁命令（仅供未来参考）
dpkg -l | grep nvidia | awk '{print $2}' | xargs sudo apt-mark unhold
```



---



设置同步文件夹+退出后不删除容器

`mkdir -p ~/docker/isaac-sim/PursuitSim3D`

`docker run --name pursuit-sim-container -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY -e "PRIVACY_CONSENT=Y" -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw -v ~/docker/isaac-sim/documents:/root/Documents:rw pursuit-sim`

新开窗口

`sudo chmod -R 755 /home/dell/docker/isaac-sim/PursuitSim3D`
`sudo chown -R $USER:$USER /home/dell/docker/isaac-sim/PursuitSim3D`

`docker cp pursuit-sim-container:/workspace/omniisaacgymenvs/PursuitSim3D ~/docker/isaac-sim/`

回到原窗口

`docker run --name pursuit-sim-container -it --gpus all -e "ACCEPT_EULA=Y" --network=host -v $HOME/.Xauthority:/root/.Xauthority -e DISPLAY -e "PRIVACY_CONSENT=Y" -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw -v ~/docker/isaac-sim/documents:/root/Documents:rw -v ~/docker/isaac-sim/PursuitSim3D:/workspace/omniisaacgymenvs/PursuitSim3D:rw pursuit-sim`

