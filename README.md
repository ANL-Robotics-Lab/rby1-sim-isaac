# rby1-sim-isaac

Rainbow Robotics **RBY1** robot simulator powered by **NVIDIA Isaac Sim 5.1.0**.

<img width="2490" height="1413" alt="image" src="https://github.com/user-attachments/assets/8a829550-300c-4841-b87f-12cef42ae295" />

## Requirements

* NVIDIA GPU + driver
* [Docker](https://docs.docker.com/engine/install/)
* [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/RainbowRobotics/rby1-sim-isaac.git
cd rby1-sim-isaac
```

### 2. Run


| Script | Robot control | Robot UDP bridge | `app_main isaac` |
|--------|---------------|------------------|------------------|
| `docker/run.sh` | Built-in standalone PD trajectory | Disabled by default | Not started |
| `docker/run_sdk.sh` | rby1-sdk / rby1 core commands | Enabled via `--udp` | Started automatically |

#### rby1-sdk UDP integration mode

Connects rby1-sdk with the Isaac Sim container to enable robot control via rby1 core.

```bash
./docker/run_sdk.sh --image 0.10.7-a_v1.2   # Model A v1.2
./docker/run_sdk.sh --image 0.10.7-m_v1.2   # Model M v1.2
```

#### Standalone mode

Runs Isaac Sim only. The robot is controlled by the built-in PD trajectory in `RBY1Task`; no `app_main` process is started and the robot command/state UDP bridge is disabled.

```bash
./docker/run.sh --image 0.10.7-a_v1.2   # Model A v1.2
./docker/run.sh --image 0.10.7-m_v1.2   # Model M v1.2
```

## Supported images

The following images are currently supported. Each image automatically loads the
USD asset matching its model (via the `ROBOT_MODEL_NAME` env var), so no extra
flag is required.

| Tag | Model |
|-----|-------|
| `0.10.7-a_v1.2` | Model A v1.2 |
| `0.10.7-m_v1.2` | Model M v1.2 |

## Docker image tag convention

```
rainbowroboticsofficial/rby1-sim-isaac:<version>-<model>_v<model_version>

e.g.) rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2   # Model A v1.2
      rainbowroboticsofficial/rby1-sim-isaac:0.10.7-m_v1.3   # Model M v1.3
```

## UDP ports

Since `--network=host` is used, the host and container share the same network namespace.
Ports `5005/5006` are used by rby1-sdk integration mode; ports `5007/5008` are used by the sim gripper bridge.

| Port | Direction | Description |
|------|-----------|-------------|
| `5005/udp` | Isaac Sim → rby1-sdk | RobotState (`ControlState`) |
| `5006/udp` | rby1-sdk → Isaac Sim | RobotCommand (`ControlInput`) |
| `5007/udp` | User code → Isaac Sim | Gripper command |
| `5008/udp` | Isaac Sim → User code | Gripper state |

## Gripper example

When Isaac Sim is running (with `--sim-gripper` enabled by default), you can directly control the gripper from the host.

```bash
cd examples
pip install numpy          # dependency

# In IsaacSim
python3 gripper_example.py --sim

# Real robot
python3 gripper_example.py
```

The `SimDynamixelBus` class in `sim_gripper_bridge.py` provides the same interface as `rby1_sdk.DynamixelBus`. By changing just one line (`--sim`), you can control both the real robot and simulation with the same code.

## License

Source files containing the NVIDIA SPDX header are licensed under Apache-2.0.
