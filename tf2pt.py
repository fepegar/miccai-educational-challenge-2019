import re
from collections import OrderedDict

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

def transpose_to_pytorch(array):
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

    
def transpose_to_tensorflow(array):
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


def niftynet_batch_to_torch_tensor(batch_dict):
    window = batch_dict['image']
    window = window[..., 0, :]  # remove time dimension
    window = transpose_to_pytorch(window)
    # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    tensor = torch.from_numpy(window.copy())
    return tensor


def torch_logits_to_niftynet_labels(logits):
    logits = logits.detach().cpu()
    labels = logits.argmax(dim=DIM_PT['channels'], keepdim=True).numpy()
    labels = labels.astype(np.uint16)
    labels = transpose_to_tensorflow(labels)
    return labels


def tf2pt_name(name_tf):
    """
    Return the equivalent PyTorch parameter name of the TensorFlow
    variable. Rules have been created from visual inspection of the
    variables lists.
    """
    param_type_dict = {
        'w': 'weight',
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean': 'running_mean',
        'moving_variance': 'running_var',
    }
    
    if name_tf.startswith('res_'):
        # For example: 'res_2_0/bn_0/moving_variance'
        pattern = (
            'res'
            r'_(\d)'  # 2 dil_idx
            r'_(\d)'  # 0 res_idx
            r'/(\w+)' # bn layer_type
            r'_(\d)'  # 0 layer_idx
            r'/(\w+)'  # moving_variance param_type
        )
        groups = re.match(pattern, name_tf).groups()
        dil_idx, res_idx, layer_type, layer_idx, param_type = groups
        param_idx = 3 if layer_type == 'conv' else 0
            
        name_pt = (
            f'block.{dil_idx}.dilation_block.{res_idx}.residual_block'
            f'.{layer_idx}.convolutional_block.{param_idx}.{param_type_dict[param_type]}'
        )
    elif name_tf.startswith('conv_'):
        conv_layers_dict = {
            'conv_0_bn_relu/conv_/w': 'block.0.convolutional_block.1.weight',  # first conv layer
            'conv_0_bn_relu/bn_/gamma': 'block.0.convolutional_block.2.weight',  
            'conv_0_bn_relu/bn_/beta': 'block.0.convolutional_block.2.bias',
            'conv_0_bn_relu/bn_/moving_mean': 'block.0.convolutional_block.2.running_mean',
            'conv_0_bn_relu/bn_/moving_variance': 'block.0.convolutional_block.2.running_var',

            'conv_1_bn_relu/conv_/w': 'block.4.convolutional_block.0.weight',  # layer with dropout
            'conv_1_bn_relu/bn_/gamma': 'block.4.convolutional_block.1.weight',  
            'conv_1_bn_relu/bn_/beta': 'block.4.convolutional_block.1.bias',
            'conv_1_bn_relu/bn_/moving_mean': 'block.4.convolutional_block.1.running_mean',
            'conv_1_bn_relu/bn_/moving_variance': 'block.4.convolutional_block.1.running_var',

            'conv_2_bn/conv_/w': 'block.6.convolutional_block.0.weight',  # classifier
            'conv_2_bn/bn_/gamma': 'block.6.convolutional_block.1.weight',  
            'conv_2_bn/bn_/beta': 'block.6.convolutional_block.1.bias',
            'conv_2_bn/bn_/moving_mean': 'block.6.convolutional_block.1.running_mean',
            'conv_2_bn/bn_/moving_variance': 'block.6.convolutional_block.1.running_var',
        }
        name_pt = conv_layers_dict[name_tf]
    return name_pt
    
    
def tf2pt(name_tf, tensor_tf):
    name_pt = tf2pt_name(name_tf)
    num_dimensions = tensor_tf.dim()
    if num_dimensions == 1:
        tensor_pt = tensor_tf
    elif num_dimensions == 5:
        tensor_pt = tensor_tf.permute(4, 3, 0, 1, 2)
    return name_pt, tensor_pt


def is_not_valid(variable_name, shape):
    exclusion_criteria = (
        'Adam' in variable_name,  # used for training
        'biased' in variable_name,  # unused
        not shape,  # empty variables
        'ExponentialMovingAverage' in variable_name,  # unused by NiftyNet model zoo
    )
    return any(exclusion_criteria)


"""
This last couple of functions are a good remainder to myself that
TensorFlow makes me sad and PyTorch makes me happy
"""
def checkpoint_to_state_dict_(checkpoint_path, csv_path, state_dict_path, filter_variables=True):
    tf.reset_default_graph()

    rows = []
    variables_dict = OrderedDict()
    variables_list = tf.train.list_variables(str(checkpoint_path))
    for name, shape in variables_list:
        if filter_variables and is_not_valid(name, shape):
            continue
        variables_dict[name] = tf.get_variable(name, shape=shape)
        name = name.replace('HighRes3DNet/', '')
        shape = ', '.join(str(n) for n in shape)
        row = {'name': name, 'shape': shape}
        rows.append(row)
    data_frame = pd.DataFrame.from_dict(rows)
    
    saver = tf.train.Saver()
    state_dict = {}
    with tf.Session() as sess:
        saver.restore(sess, str(checkpoint_path))
        for name, shape in variables_list:
            if filter_variables and is_not_valid(name, shape):
                continue
            array = variables_dict[name].eval()
            name = name.replace('HighRes3DNet/', '')
            state_dict[name] = torch.tensor(array)
    
    data_frame.to_csv(csv_path)
    torch.save(state_dict, state_dict_path)


def checkpoint_to_state_dict(*args, **kwargs):
    """
    https://stackoverflow.com/a/44842044/3956024
    """
    import multiprocessing
    p = multiprocessing.Process(
        target=checkpoint_to_state_dict_,
        args=args,
        kwargs=kwargs,
    )
    p.start()
    p.join()
    