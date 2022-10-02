import unittest
import torch
import pprint

from tcn_block import tConv1d
from tcn_block.misc import gridGenerator, pick_dict_values


class tConvTests(unittest.TestCase):
    argument_variations = {
        **(tConv1d.allowed_input_values),
        'in_channels': [2, 4, 10],
        'kernel_size': [2, 3, 5, 10],
        'in_size': [35, 100, 300],
        'output_size': [1, 2, 3, 5],
    }

    def test_module_creation(self):
        default_args = {'in_channels': 10, 'kernel_size': 2,
                        'in_size': 20, 'output_size': 5, }
        for vals in gridGenerator(pick_dict_values(tConvTests.argument_variations,
                                                   ['consumption', 'normalization', 'output',
                                                    'residual', 'residual_conv',
                                                    'skip_connections', 'skip_conv'])):
            input_args = {**vals, **default_args}
            _ = tConv1d(**input_args)

    def test_input_size(self):
        in_size = 10

        _ = tConv1d(in_channels=10, kernel_size=2,
                    in_size=in_size, output_size=in_size-1)

        for outp in [in_size, in_size+1]:
            with self.assertRaises(ValueError):
                _ = tConv1d(in_channels=10, kernel_size=2,
                            in_size=in_size, output_size=outp)

    def test_output_size(self):
        channels = 2
        output_size = 5
        for vals in gridGenerator(pick_dict_values(tConvTests.argument_variations,
                                                   ['in_size', 'kernel_size',
                                                    'consumption', 'normalization',
                                                    'residual', 'residual_conv',
                                                    'skip_connections', 'skip_conv', ])):
            conv = tConv1d(**{**vals, 'in_channels': channels,
                              'output_size': output_size})

            sample = torch.rand(1, channels, vals['in_size'])

            try:
                res = conv(sample)
            except Exception as e:
                print(vals)
                raise e

            self.assertEqual(int(res.size()[-1]), output_size)
