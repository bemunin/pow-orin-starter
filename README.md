# Pow Orin Starter

A starter template for streamlined NVIDIA Isaac Sim and Isaac ROS workflows on Jetson Orin devices, powered by the Isaac Powerpack CLI (`pow-cli`).

**Current Features include:**
- Quick Isaac Sim installation using a Isaac Sim Python environment and uv
- Initialize Isaac Sim v5.1.0 with a single command
- Simple and expressive configuration via `pow.toml`
- Manage multiple Isaac Sim application contexts using `pow.toml` profiles
- Easily download and install Isaac Sim local assets

🚧 This starter repository currently uses the `pow-cli` alpha version `0.1.0a1`, which is under early development.

📌 For more information and the latest updates, please refer to the [Isaac Powerpack GitHub repository](https://github.com/bemunin/isaac-powerpack).

## 🚀 Quick Start

1. Install uv by following the offical instructions [here](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
   
2. Install isaacsim 5.1.0 and the pow-cli pre-release package:
   ```bash
   # Create the environment and install packages
   uv sync --extra sim
   
   # Activate the project virtual environment
   source ./.venv/bin/activate

   ```

3. Check system compatibility:
   ```bash
   # Accept the NVIDIA EULA agreement by typing Yes when prompted
   pow sim check 
   ```
3. Initialize sim project:
   ```bash
   # Initialize project, Accept the NVIDIA EULA agreement by typing YES when prompted
   pow sim init
   ```
4. Run isaac sim (see configuration section to customize launch settings):
   ```bash
   # Run with the default pow.toml profile
   pow sim run

   # Or run with a specific profile
   pow sim run -p [PROFILE_NAME]
   ```

5. (Optional) Install local assets
   
   Stop Isaac Sim if it is running, then execute:
   ```bash
   pow sim add local-assets [ASSET_ROOT]
   ```
   This command will download and install the local assets to the specified `ASSET_ROOT` path. 

   This process may take some time. You can stop and resume the download at any time. The command will automatically detect unfinished downloads and resume from where it left off.

   Check the local assets usage status:
   ```bash
   pow sim info --local-assets
   # or
   pow sim info -l
   ```   

   Finally, run Isaac Sim again with `--reset-user` flag to update the kit asset config:
   ```bash
   pow sim run -- --reset-user
   ```   

## ⚙️ Configuration

You can configure simulation init and launch by editing the `pow.toml` file.  Both `pow sim init` and `pow sim run` command will load the configuration defined in the `pow.toml`.


### Simulation Settings
sim section will set broad simulation settings
```toml
[sim]
# isaacsim version (currently support only 5.1.0)
version = "5.1.0"
# add paths to additional extension folders
ext_folders = ["./sim/exts"]
```
in `ext_folders` you can add paths to additional extension folders that you want to include in the simulation environment.

### Isaac Sim ROS workspace settings

This section will enable Isaac Sim ROS workspace that allows seamless rclpy integration for isaacsim.ros2.bridge extension.

```toml
[sim.ros]
enable_ros = true
isaacsim_ros_ws = "~/.pow/IsaacSim-ros_workspaces"
# support: humble, jazzy
ros_distro = "humble"
```

during `pow sim init`, if `enable_ros` is set to true, the command will download and build the Isaac Sim ROS workspace for the specified `ros_distro` in `~/.pow` directory.

during `pow sim run`, if `enable_ros` is true, the command will source the ROS workspace setup file before launching Isaac Sim.

### Profiles
You can define multiple profiles in the `pow.toml` file to manage different simulation contexts. Each profile can have its own settings that override the default settings.

to run profile selectively, you can use:
```bash
# run sim with specific profile
pow sim run --profile [PROFILE_NAME]

# short form
pow sim run -p [PROFILE_NAME]


```

currently supported profile settings are:

```toml
[[sim.profiles]]
name = "default"
cpu_performance_mode = true
headless = false
extensions = ["isaacsim.code_editor.vscode"]
raw_args = ["--/renderer/raytracingMotion/enabled=false"]
open_scene_path = ""
```

- **name**: profile name, this is requied field to identify the profile in the `pow sim run` command
- **cpu_performance_mode**: enable cpu performance mode

- **headless**: run isaac sim in headless mode
- **extensions**: list of extensions to load during isaac sim launch
- **raw_args**: list of raw isaacsim command line arguments to pass to isaac sim during launch 
- **open_scene_path**: path to the scene file to open during isaac sim launch



To create a new profile, simply add a new `[[sim.profiles]]` section in the `pow.toml` file with a unique name and desired settings.

```toml
[[sim.profiles]]
name = "my_custom_profile"
cpu_performance_mode = false
headless = true
extensions = ["isaacsim.some_other_extension"]
```

It will extend the profile named 'default'. Any field that is specified in the new profile will override the corresponding field in the 'default' profile.