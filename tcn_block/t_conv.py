import torch.nn as nn
import torch.nn.functional as F
from .misc import switch_norm1d, get_input_args
from .dilated_layer.dilated_layer import configurable_dilated_layer


class t_conv(nn.Module):

    def __init__(self,
                 in_channels, kernel_size,
                 in_size, output_size,
                 consumption, normalization='batch', output="default",
                 residual=False, residual_conv=False,
                 skip_connections=False, skip_conv=False,
                 debug=False
                 ):
        super().__init__()

        if consumption not in ['trim', 'padded', 'full']:
            consumption = 'full'
        if normalization not in [None, 'batch', 'switch']:
            normalization = 'batch'
        if output not in ['default', 'double']:
            output = 'default'
        if (residual_conv):
            residual = True
        if (skip_conv):
            skip_connections = True
        input_args = get_input_args(locals())

        layer_guide = self._calculate_layer_array(
            in_size, output_size, kernel_size, consumption)

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
        flag = 0
        if hasattr(self, 'trim'):
            remaining -= self.trim
            flag += 1
        if hasattr(self, 'pad'):
            remaining += self.pad
            flag += 1
        if flag > 1:
            raise ValueError(f"Wtf just happened? pad + trim found")
        norm_array = []
        for lg in layer_array:
            remaining -= lg[0]**lg[1] * (lg[0] - 1)
            if normalization == 'batch':
                norm_array.append(nn.BatchNorm1d(in_channels))
            elif normalization == 'switch':
                norm_array.append(switch_norm1d(in_channels, remaining))
        return norm_array

    def _calculate_layer_array(self, in_size, output_size, kernel_size, consumption):
        layer_array = []
        remaining_input = in_size
        while True:
            layers_num = self._get_max_layers(
                remaining_input - (output_size - 1), kernel_size)

            if consumption == 'padded':
                layers_num += 1

            for i in range(layers_num):
                layer_array.append([kernel_size, i])

            if consumption == 'padded':
                self.pad = self._get_receptive_field(
                    kernel_size, layers_num) + (output_size - 1) - in_size
                break
            elif consumption == 'trim':
                self.trim = in_size - \
                    (self._get_receptive_field(
                        kernel_size, layers_num) + (output_size - 1))
                break
            elif consumption == 'full':
                if layers_num == 0:
                    k = remaining_input - (output_size - 1)
                    if k == 1:
                        break
                    layer_array.append([k, 0])

                    break
                remaining_input = remaining_input\
                    - (self._get_receptive_field(kernel_size, layers_num) - 1)

        return layer_array

    @staticmethod
    def _get_receptive_field(kernel: int, layers: int) -> int:
        return kernel ** layers

    @staticmethod
    def _get_max_layers(input_size: int, kernel: int) -> int:
        i = 0
        while True:
            if t_conv._get_receptive_field(kernel, i + 1) > input_size:
                break
            i += 1
        return i
