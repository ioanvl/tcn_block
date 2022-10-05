import torch.nn as nn
import torch.nn.functional as F
from .misc import switch_norm1d, get_input_args
from .dilated_layer.dilated_layer import configurable_dilated_layer


class TConv1d(nn.Module):

    allowed_input_values = {
        'consumption': ['trim', 'padded', 'full'],
        'normalization': [None, 'batch', 'switch'],
        'output': ['default', 'double'],
        'residual': [True, False, 'conv'],
        'skip_connections': [True, False, 'conv']
    }

    def __init__(self,
                 in_channels, kernel_size,
                 in_size, output_size,
                 consumption='full', normalization='batch', output='default',
                 residual=False, skip_connections=False,
                 debug=False
                 ):
        super().__init__()

        if output_size >= in_size:
            raise ValueError(
                F"Output size must be smaller than input: \nin:[{in_size}], out:[{output_size}]")
        if (consumption == 'trim') and ((in_size - (kernel_size - 1)) < output_size):
            raise ValueError(
                F"With \"trim\" consumption the input size must be at least equal to \
[output_size + kernel_size - 1] or larger \n[in]:{in_size}, [kernel]:{kernel_size}, [out]:{output_size}")

        for k in TConv1d.allowed_input_values:
            if locals()[k] not in TConv1d.allowed_input_values[k]:
                raise ValueError(F"Invalid argument for [{k}] = {locals()[k]}, \
alllowed values: {TConv1d.allowed_input_values[k]}")
        input_args = get_input_args(locals())

        layer_guide = self._calculate_layers(
            in_size, output_size, kernel_size, consumption)

        if consumption == 'padded':
            self.pad = self._get_receptive_field(
                kernel_size, len(layer_guide)) + (output_size - 1) - in_size

        elif consumption == 'trim':
            self.trim = in_size - \
                (self._get_receptive_field(
                    kernel_size, len(layer_guide)) + (output_size - 1))

        if debug:
            print(layer_guide)

        layers = []
        for i, lg in enumerate(layer_guide):
            h = configurable_dilated_layer(
                **{**input_args, 'kernel_size': lg[0], 'dilation': (lg[0]**lg[1])})
            h.name = f"l{i}k{kernel_size} - Dil: {lg[0]**lg[1]}"
            if debug:
                print(h.name)
            layers.append(h)
        self.layers = nn.ModuleList(layers)

        if normalization:
            norms = self._create_norm_array(
                in_channels, in_size, normalization, layer_guide)
            self.norms = nn.ModuleList(norms)

        self._calculate_forward(
            consumption, normalization, output, skip_connections)

    def _calculate_forward(self, consumption,
                           normalization, output, skip_connections):
        if normalization:
            self.f_norm = self._norm_pass
        else:
            self.f_norm = self._passthrough
        if skip_connections:
            if output == 'default':
                self.forward = self._skip_single_out
            else:
                self.forward = self._skip_forward
        else:
            self.forward = self._simple_forward

        if consumption == 'trim':
            self.preproc = self._preproc_trim
        elif consumption == 'padded':
            self.preproc = self._preproc_pad
        elif consumption == 'full':
            self.preproc = self._passthrough

    def _preproc_trim(self, x):
        return x.narrow(-1, 0, (x.size()[-1] - self.trim))

    def _preproc_pad(self, x):
        return F.pad(input=x, pad=(0, self.pad, 0, 0, 0, 0), mode='constant', value=0)

    def _norm_pass(self, x, i):
        return self.norms[i](x)

    def _passthrough(self, x, *args, **kwargs):
        return x

    def _simple_forward(self, x):
        x = self.preproc(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.f_norm(x, i)
        return x

    def _skip_forward(self, x):
        x = self.preproc(x)
        x, skips = self.layers[0](x)
        x = self.f_norm(x, 0)
        for i, layer in enumerate(self.layers[1:]):
            x, skip = layer(x)
            x = self.f_norm(x, i+1)
            skips += skip
        return x, skips

    def _skip_single_out(self, x):
        x, skips = self._skip_forward(x)
        return skips

    def _create_norm_array(self, in_channels, in_size, normalization, layer_array):
        remaining = in_size
        if hasattr(self, 'trim'):
            remaining -= self.trim
        if hasattr(self, 'pad'):
            remaining += self.pad

        norm_array = []
        for lg in layer_array:
            remaining -= lg[0]**lg[1] * (lg[0] - 1)
            if normalization == 'batch':
                norm_array.append(nn.BatchNorm1d(in_channels))
            elif normalization == 'switch':
                norm_array.append(switch_norm1d(in_channels, remaining))
        return norm_array

    @staticmethod
    def _calculate_layers(in_size, output_size, kernel_size, consumption):
        layer_array = []
        remaining_input = in_size
        while True:
            layers_num = TConv1d._get_max_layers(
                remaining_input - (output_size - 1), kernel_size)

            if consumption == 'padded':
                layers_num += 1

            for i in range(layers_num):
                layer_array.append([kernel_size, i])

            if consumption in ['padded', 'trim']:
                break

            if consumption == 'full':
                if layers_num == 0:
                    k = remaining_input - (output_size - 1)
                    if k == 1:
                        break
                    layer_array.append([k, 0])

                    break
                remaining_input = remaining_input\
                    - (TConv1d._get_receptive_field(kernel_size, layers_num) - 1)

        return layer_array

    @staticmethod
    def _get_receptive_field(kernel: int, layers: int) -> int:
        return kernel ** layers

    @staticmethod
    def _get_max_layers(input_size: int, kernel: int) -> int:
        i = 0
        while True:
            if TConv1d._get_receptive_field(kernel, i + 1) > input_size:
                break
            i += 1
        return i
