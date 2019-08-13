from pathlib import Path
from collections import OrderedDict
from typing import Callable, Tuple, Optional, Union

import torch
import numpy as np
import pandas as pd
import tensorflow as tf


DIM_NN = {
    'batch': 0,
    'depth': 1,
    'height': 2,
    'width': 3,
    'channels': 4,
}

DIM_PT = {
    'batch': 0,
    'channels': 1,
    'depth': 2,
    'height': 3,
    'width': 4,
}


def transpose_to_pytorch(array: np.ndarray) -> np.ndarray:
    """
    See docs of torch.nn.Conv3d
    https://pytorch.org/docs/stable/nn.html#conv3d
    """
    shape = (
        DIM_NN['batch'],
        DIM_NN['channels'],
        DIM_NN['depth'],
        DIM_NN['height'],
        DIM_NN['width'],
    )
    array = np.transpose(array, shape)
    return array


def transpose_to_tensorflow(array: np.ndarray) -> np.ndarray:
    """
    See docs of tf.nn.conv3d
    https://www.tensorflow.org/api_docs/python/tf/nn/conv3d#args
    """
    shape = (
        DIM_PT['batch'],
        DIM_PT['depth'],
        DIM_PT['height'],
        DIM_PT['width'],
        DIM_PT['channels'],
    )
    array = np.transpose(array, shape)
    return array


def niftynet_batch_to_torch_tensor(batch_dict: dict) -> torch.Tensor:
    window = batch_dict['image']
    window = window[..., 0, :]  # remove time dimension
    window = transpose_to_pytorch(window)
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    tensor = torch.from_numpy(window.copy())
    return tensor


def torch_logits_to_niftynet_labels(logits: torch.Tensor) -> np.ndarray:
    logits = logits.detach().cpu()
    labels = logits.argmax(dim=DIM_PT['channels'], keepdim=True).numpy()
    labels = labels.astype(np.uint16)
    labels = transpose_to_tensorflow(labels)
    return labels


def tf2pt(
        name_tf: str,
        array_tf: np.ndarray,
        name_mapping_function: Callable,
        ) -> Tuple[str, np.ndarray]:
    name_pt = name_mapping_function(name_tf)
    num_dimensions = array_tf.dim()
    if num_dimensions == 1:
        array_pt = array_tf
    elif num_dimensions == 5:
        array_pt = array_tf.permute(4, 3, 0, 1, 2)
    else:
        raise NotImplementedError
    return name_pt, array_pt


"""
This last couple of functions are a good reminder to myself that
TensorFlow makes me sad and PyTorch makes me happy
"""


def checkpoint_tf_to_state_dict_tf_(
        input_checkpoint_tf_path: Union[str, Path],
        output_csv_tf_path: Union[str, Path],
        output_state_dict_tf_path: Union[str, Path],
        filter_out_function: Optional[Callable] = None,
        replace_string: Optional[str] = None,
        ) -> None:
    tf.reset_default_graph()

    rows = []
    variables_dict = OrderedDict()
    variables_list = tf.train.list_variables(str(input_checkpoint_tf_path))
    for name, shape in variables_list:
        if filter_out_function is not None and filter_out_function(name, shape):
            continue
        variables_dict[name] = tf.get_variable(name, shape=shape)
        if replace_string is not None:
            name = name.replace(replace_string, '')
        shape = ', '.join(str(n) for n in shape)
        row = {'name': name, 'shape': shape}
        rows.append(row)
    data_frame = pd.DataFrame.from_dict(rows)

    saver = tf.train.Saver()
    state_dict = {}
    with tf.Session() as sess:
        saver.restore(sess, str(input_checkpoint_tf_path))
        for name, shape in variables_list:
            if (
                    filter_out_function is not None
                    and filter_out_function(name, shape)
                    ):
                continue
            array = variables_dict[name].eval()
            if replace_string is not None:
                name = name.replace(replace_string, '')
            state_dict[name] = torch.tensor(array)

    data_frame.to_csv(output_csv_tf_path)
    torch.save(state_dict, output_state_dict_tf_path)


def checkpoint_tf_to_state_dict_tf(*args, **kwargs) -> None:
    """
    This is done so that the GPU can be used by PyTorch afterwards
    https://stackoverflow.com/a/44842044/3956024

    If you don't need to run the inference, call
    checkpoint_tf_to_state_dict_tf_ instead to avoid potential headaches

    This might break things on Windows if trying to run as a Python file.
    See https://github.com/pytorch/pytorch/issues/5858#issuecomment-373950687
    """
    import multiprocessing
    p = multiprocessing.Process(
        target=checkpoint_tf_to_state_dict_tf_,
        args=args,
        kwargs=kwargs,
    )
    p.start()
    p.join()
