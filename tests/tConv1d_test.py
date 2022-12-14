import unittest
import torch
import pprint

from tcn_block import TConv1d
from tcn_block.misc import gridGenerator, pick_dict_values


argument_variations = {
    **(TConv1d.allowed_input_values),
    'in_channels': [2, 4, 10],
    'kernel_size': [2, 3, 5, 10],
    'in_size': [10, 35, 100, 300],
    'output_size': [1, 2, 3, 5, 10],
}

argument_defaults = {
    'in_channels': 8,
    'kernel_size': 3,
    'in_size': 20,
    'output_size': 5,
    'consumption': 'full',
    'normalization': 'batch',
    'output': 'default',
    'residual': False,
    'skip_connections': False,
    'debug': False,
}


class tConvTests(unittest.TestCase):

    def test_module_creation(self):
        default_args = {'in_channels': 10, 'kernel_size': 2,
                        'in_size': 20, 'output_size': 5, }
        for vals in gridGenerator(pick_dict_values(argument_variations,
                                                   ['consumption', 'normalization', 'output',
                                                    'residual', 'residual_conv',
                                                    'skip_connections', 'skip_conv'])):
            input_args = {**vals, **default_args}
            _ = TConv1d(**input_args)

    def test_input_size(self):
        in_size = 10

        _ = TConv1d(in_channels=10, kernel_size=2,
                    in_size=in_size, output_size=in_size-1)

        with self.assertRaises(ValueError):
            _ = TConv1d(in_channels=10, kernel_size=3,
                        in_size=in_size, output_size=in_size-1, consumption='trim')

        for outp in [in_size, in_size+1]:
            with self.assertRaises(ValueError):
                _ = TConv1d(in_channels=10, kernel_size=2,
                            in_size=in_size, output_size=outp)

    def test_output_size(self):
        channels = 2
        output_size = 5
        for vals in gridGenerator(pick_dict_values(argument_variations,
                                                   ['in_size', 'kernel_size',
                                                    'consumption', 'normalization',
                                                    'residual', 'residual_conv',
                                                    'skip_connections', 'skip_conv', ])):
            if ((output_size >= vals['in_size']) or
                    (vals['consumption'] == 'trim') and
                    ((vals['in_size'] - (vals['kernel_size'] - 1)) < output_size)):

                with self.assertRaises(ValueError):
                    _ = TConv1d(**{**vals, 'in_channels': channels,
                                   'output_size': output_size})
                continue

            conv = TConv1d(**{**vals, 'in_channels': channels,
                                      'output_size': output_size})
            sample = torch.rand(1, channels, vals['in_size'])

            try:
                res = conv(sample)
            except Exception as e:
                print(vals)
                raise e

            self.assertEqual(int(res.size()[-1]), output_size)
