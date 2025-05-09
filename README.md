# Augmenting flexibility and autonomy of intellegent robots with Behavior Primitives for Diverse Manipulation Tasks

This is the codebase for learning primitive skills and a try of reusing them with cognitive archiectures for enabling the robots with capabilities of adaptability and flexibility in new environment and tasks.

This guide contains information about (1) [Installation](#installation), (2) [Running Experiments](#running-experiments), (3) [Setting Up Your Own Environments](#setting-up-your-own-environments), (4) [Acknowledgement](#acknowledgement), and (5) [Citation](#citation).

## Installation
### Download code
- Current codebase: ```git clone https://github.com/xuejianyong/primitive_skills_learning```
- (for environments) the `maple` branch in robosuite: ```git clone -b maple https://github.com/ARISE-Initiative/robosuite```

### Setup robosuite 
1. Download [MuJoCo 2.0](https://gist.github.com/ellisbrown/47bfd3e524aed11216cd3c0a0872a654) (Linux and Mac OS X) and unzip its contents into `~/.mujoco/mujoco200`, and copy your MuJoCo license key `~/.mujoco/mjkey.txt`. You can obtain a license key from [here](https://www.roboti.us/license.html).
2. (linux) Setup additional dependencies: ```sudo apt install libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools xpra xserver-xorg-dev libglfw3-dev patchelf```
3. Add MuJoCo to library paths: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin`

### Setup conda environment
1. Create the conda environment: `conda env create --name ppl --file=maple.yml`
2. (if above fails) edit `ppl.yml` to modify dependencies and then resume setup: `conda env update --name maple --file=maple.yml`
3. Activate the conda environment: `conda activate maple`
4. Finish maple setup: (in your maple repo path do) `pip install -e .`
5. Finish robosuite setup: (in your robosuite repo path do) `pip install -e .`

## Running Experiments
Scripts for training policies and re-playing policy checkpoints are located in `scripts/train.py` and `scripts/eval.py`, respectively.

These experiment scripts use the following structure:
```
base_variant = dict(
  # default hyperparam settings for all envs
)

env_params = {
  '<env1>' : {
    # add/override default hyperparam settings for specific env
    # each setting is specified as a dictionary address (key),
    # followed by list of possible options (value).
    # Example in following line:
    # 'env_variant.controller_type': ['OSC_POSITION'],
  },
  '<env2>' : {
    ...
  },
}
```

### Command Line Options
See `parser` in `scripts/train.py` for a complete list of options. Some notable options:
- `env`: the env to run (eg. `stack`)
- `label`: name for experiment
- `debug`: run with lite options for debugging

### Plotting Experiment Results
During training, the results will be saved to a file called under `LOCAL_LOG_DIR/<env>/<exp_prefix>/<foldername>`.
Inside this folder, the experiment results are stored in `progress.csv`. We recommend using [viskit](https://github.com/vitchyr/viskit) to plot the results.

## Setting Up Your Own Environments
Note that this codebase is designed to work with robosuite environments only. For setting up your own environments, please follow [these examples](https://github.com/ARISE-Initiative/robosuite/tree/maple/robosuite/environments/manipulation) for reference. Notably, you will need to add the `skill_config` variable to the constructor, and define the keypoints for the affordance score by implementing the `_get_skill_info` function.

If you would like to know the inner workings of the primitives, refer to [`skill_controller.py`](https://github.com/ARISE-Initiative/robosuite/tree/maple/robosuite/controllers/skill_controller.py) and [`skills.py`](https://github.com/ARISE-Initiative/robosuite/tree/maple/robosuite/controllers/skills.py). Note that we use the term "skill" to refer to behavior primitives in the code.

## Acknowledgement
Much of this codebase is directly based on [RLkit](https://github.com/vitchyr/rlkit), which itself is based on [rllab](https://github.com/rll/rllab).
In addition, the environments were developed as a forked branch of [robosuite](https://github.com/ARISE-Initiative/robosuite) `v1.1.0`.

## Citation
