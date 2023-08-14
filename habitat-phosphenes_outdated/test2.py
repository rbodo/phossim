import os
import sys
from pathlib import Path

from habitat_baselines.run import execute_exp
# Need this import to register custom transformers.
import phosphenes

def main():
    # DISPLAY =:11.0
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    os.chdir(Path('~/Internship/PyCharm_projects/habitat-lab/').expanduser())

    path_config = str(Path('~/Internship/PyCharm_projects/Phossim/habitat-phosphenes/'
                           'ppo_pointnav_phosphenes_complete.yaml').expanduser())

    _config = phosphenes.get_config(path_config)

    execute_exp(_config, 'eval') #train or eval


if __name__ == '__main__':
    main()

    sys.exit()
