# rby1-sim-isaac

Rainbow Robotics **RBY1** robot simulator powered by **NVIDIA Isaac Sim 5.1.0**.

<img width="2267" height="1275" alt="RBY1 robot in Isaac Sim" src="docs/Isaac_sim_scene.png?v=3" />

## Requirements

* NVIDIA GPU + driver
* [Docker](https://docs.docker.com/engine/install/)
* [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Minimal install check

```bash
nvidia-smi
docker --version
dpkg -l | grep -E 'nvidia-container-toolkit|nvidia-container-runtime'
```

If `nvidia-smi` prints the GPU table and the last command lists an NVIDIA
container package, the NVIDIA driver, Docker, and NVIDIA Container Toolkit are
installed.

## Quick start

### Windows native rendering

The Isaac Sim container requires a native Linux host. On Windows, run
`src/simulation.py` with a native Isaac Sim Python environment and use D3D12:

```powershell
python src/simulation.py --model a --graphics-api d3d12
```

`d3d12` is the default on Windows; the explicit option is shown here to make
the selected backend clear in logs. Linux and the Docker image continue to
default to Vulkan. Native Windows also uses the portable UDP codec in `src/`;
the vendor image continues to use its bundled Linux codec.

For SDK integration on Windows, keep Isaac Sim native and run only the SDK
backend in WSLg. The bundled `app_main` creates a small CImg status window, so
the backend container still needs access to WSLg's X11 display:

```bash
# WSL (Ubuntu-22.04)
xhost +SI:localuser:\#1234 +local:
sudo docker run --rm -it \
  --name rby1-sdk-backend \
  --network host \
  --env DISPLAY="${DISPLAY:-:0}" \
  --env XAUTHORITY=/isaac-sim/.Xauthority \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --volume "$HOME/.Xauthority:/isaac-sim/.Xauthority:rw" \
  --user 1234:1234 \
  --entrypoint /opt/rby1-sim-isaac/sdk/app/app_main \
  rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2 \
  isaac
```

Then start the native D3D12 simulator in Windows PowerShell:

```powershell
python src/simulation.py --model a --graphics-api d3d12 --udp `
  --state-ip 127.0.0.1 --state-port 5005 --cmd-port 5006
```

Do not use `docker/run_sdk.sh` for this split configuration: that launcher also
starts the Linux Isaac Sim renderer and therefore selects Vulkan.

#### Replay a generated motion-library PKL

Use a separate Windows environment for the SDK client because RBY1 SDK 0.9.1
requires NumPy 2, while the Isaac Sim environment currently uses NumPy 1.26:

```powershell
# Run from the GR00T-WholeBodyControl repository root.
python -m venv rby1_hardware\.venv-sdk
.\rby1_hardware\.venv-sdk\Scripts\python.exe -m pip install rby1-sdk==0.9.1 joblib
```

With the WSL SDK backend and native D3D12 simulator still running, start the
dance from a third Windows PowerShell terminal:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sdk_replay.py `
  .\tmp_rby1_video_compare\guy_dancing_rby1_from_original_bvh_trainable_9fps_smoothed7.pkl `
  --target simulator `
  --address 127.0.0.1:50051 `
  --auto-enable `
  --fit-start-pose `
  --speed 0.20 `
  --torso-scale 0.40 `
  --arm-scale 1.0 `
  --head-scale 0.25 `
  --transition-seconds 8 `
  --countdown 5
```

Wheels remain disabled unless `--enable-wheels` is supplied. The runner first
transitions the one-sided shoulder joints to a fitted baseline, then replays
the original motion deltas. Warnings under `Original full-amplitude PKL` are
diagnostic; execution proceeds only if `Measured-start replay trajectory`
passes the configured URDF limits. The transition is streamed at the requested
control rate and continues on the same SDK command stream as the motion. This is
required by the split Windows-Isaac/WSL-SDK backend, whose one-shot position
command handler may remain in `Executing` after the simulated joints settle.
The conservative `0.20` playback speed also leaves margin under the backend's
own approximately 0.20-rad joint tracking-fault threshold. If a replay aborts,
press Backspace in Isaac Sim to reset to the zero pose before using
`--fit-start-pose` again; the runner refuses to stack fitted offsets on an
aborted pose.

The command above is a conservative measured-pose-relative hardware trial. To
compare against GROOT/IsaacLab's kinematic replay, reset Isaac Sim and use the
PKL's absolute upper-body pose at a slower rate:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sdk_replay.py `
  .\tmp_rby1_video_compare\guy_dancing_rby1_from_original_bvh_trainable_9fps_smoothed7.pkl `
  --target simulator --address 127.0.0.1:50051 --auto-enable `
  --reference-mode absolute --fit-start-pose `
  --speed 0.10 --torso-scale 1 --arm-scale 1 --head-scale 1 `
  --transition-seconds 12 --stationary-timeout 8 --countdown 5
```

The vendor URDF has tighter limits than the clean motion model. Fitting therefore
redistributes constant corrections across the serial torso joints so total
torso pitch/roll remains unchanged, while making small wrist-pitch corrections.
IsaacLab teleports the floating root exactly; SDK replay cannot do that and keeps
the mobile base fixed unless wheels are explicitly enabled. After confirming the
absolute upper-body motion, add `--enable-wheels --wheel-scale 1` in the simulator
to approximate the PKL's planar base path.

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
./docker/run_sdk.sh --image 0.10.7-a_v1.2 --gripper --gripper-name rb_gripper
./docker/run_sdk.sh --image 0.10.7-m_v1.2 --gripper --gripper-name rb_gripper
```

#### Standalone mode

Runs Isaac Sim only. The robot is controlled by the built-in PD trajectory in `RBY1Task`; no `app_main` process is started and the robot command/state UDP bridge is disabled.

```bash
./docker/run.sh --image 0.10.7-a_v1.2   # Model A v1.2
./docker/run.sh --image 0.10.7-m_v1.2   # Model M v1.2
GRIPPER_NAME=rb_gripper ./docker/run.sh --image 0.10.7-m_v1.2
```

The default launch uses the no-gripper base USD. Additional arguments after
`--image` are forwarded to `simulation.py`, including `--gripper`,
`--gripper-name <name>`, and `--no-sim-gripper`. A gripper can also be selected
and enabled with the `GRIPPER_NAME` environment variable.

### Custom gripper assets

The simulator loads gripper assets by folder name, so a different hand can be
attached without modifying the base RBY1 USD. Place the hand USD files and mount
configuration under `assets/gripper/<gripper-name>/`:

<img width="1280" height="900" alt="RBY1 custom gripper attachment structure" src="docs/gripper_attachment_overview.svg?v=2" />

```text
assets/gripper/<gripper-name>/
├── <gripper-name>_left.usd
├── <gripper-name>_right.usd
└── gripper.json
```

`gripper.json` defines which prim in each hand is fixed to the robot wrist:

```json
{
  "left": {
    "body": "link_mount_left",
    "mount_pos": [0.0, 0.0, -0.1261],
    "mount_rot": [0.0, 0.70710678, -0.70710678, 0.0]
  },
  "right": {
    "body": "link_mount_right",
    "mount_pos": [0.0, 0.0, -0.1261],
    "mount_rot": [0.0, 0.70710678, 0.70710678, 0.0]
  }
}
```

For example, a user-provided hand can be tested by placing USD files under
`assets/gripper/custom_gripper/` and launching:

```bash
./docker/run.sh --image 0.10.7-a_v1.2 --gripper --gripper-name custom_gripper
```

User-provided USD files are not included in this repository; only the attachment
pattern and `gripper_servers/custom_gripper.py` adapter skeleton are provided.

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
