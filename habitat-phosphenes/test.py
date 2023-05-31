import habitat # To avoid the problem with libllvmlite
import os
import sys
from pathlib import Path
# Config management (change parameters from script)
# from omegaconf import OmegaConf
# import yaml

from habitat_baselines.run import execute_exp
# Need this import to register custom transformers.
import phosphenes

def repeat_process():
    path_config = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_test.yaml').expanduser())

    variable_list = [2, 4, 8]
    for variable in variable_list:
        # Load the YAML file into a Python object
        with open(path_config, "r") as f:
            config = yaml.safe_load(f)

        # Modify the value of an existing field in the third embedded dictionary
        config["habitat_baselines"][
            "tensorboard_dir"] = "/home/carsan/Data/phosphenes/habitat/tb/phosphenes" + "/ppo_epoch" + str(variable)
        config["habitat_baselines"]["rl"]["ppo"]["ppo_epoch"] = variable

        # Write the modified Python object back to the YAML file
        with open(path_config, "w") as f:
            yaml.dump(config, f)

        _config = phosphenes.get_config(path_config)

        execute_exp(_config, 'train')

def main():
    # DISPLAY =:11.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '9'

    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())

    path_config = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_complete.yaml').expanduser())

    _config = phosphenes.get_config(path_config)

    # config_dict = OmegaConf.to_container(_config, resolve=True)
    # path_config2 = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
    #                        'ppo_pointnav_phosphenes_complete.yaml').expanduser())
    # with open(path_config2, "w") as f:
    #     yaml.dump(config_dict, f)

    execute_exp(_config, 'train')


if __name__ == '__main__':
    main()

    sys.exit()
