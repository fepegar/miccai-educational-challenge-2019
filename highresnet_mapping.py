import re
from typing import Tuple


def is_not_valid(variable_name: str, shape: Tuple[int, ...]) -> bool:
    exclusion_criteria = (
        'Adam' in variable_name,  # used for training
        'biased' in variable_name,  # unused
        not shape,  # empty variables
        'ExponentialMovingAverage' in variable_name,  # unused on NiftyNet model zoo
    )
    return any(exclusion_criteria)


def tf2pt_name(name_tf: str) -> str:
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
            r'_(\d)'   # dil_idx = 2
            r'_(\d)'   # res_idx = 0
            r'/(\w+)'  # layer_type = bn
            r'_(\d)'   # layer_idx = 0
            r'/(\w+)'  # param_type = moving_variance
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
    else:
        raise NotImplementedError
    return name_pt
