import os
from typing import Union
from pathlib import Path

import numpy as np
from niftynet.io.image_reader import ImageReader


def list_files(startpath: Union[str, Path]) -> None:
    """
    Adapted from https://stackoverflow.com/a/9728478/3956024
    """
    startpath = str(startpath)
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


def get_first_array(directory: Union[str, Path]) -> np.ndarray:
    """
    Use NiftyNet's reader to get RAS images
    """
    directory = Path(directory)
    image_dir = list(directory.glob('**/*.nii.gz'))[0].parent
    input_dict = dict(
        path_to_search=str(image_dir),
        filename_contains='nii',
        axcodes=('R', 'A', 'S'),
    )
    data_parameters = {
        'image': input_dict,
    }
    reader = ImageReader().initialise(data_parameters)
    _, image_dict, _ = reader()
    return image_dict['image'].squeeze()
