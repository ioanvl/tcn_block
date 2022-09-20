import torch.nn as nn
from misc import switch_norm1d, get_input_args
from dilated_layer.dilated_layer import configurable_dilated_layer


class t_conv(nn.Module):

    def __init__(self,
                 in_channels, kernel_size,
                 in_size, output_size,
                 consumption, normalization, output,
                 residual=False, residual_conv=False,
                 skip_connections=False, skip_conv=False,
                 debug=False
                 ):
        super().__init__()

        if consumption not in [None, 'minimum', 'padded', 'full']:
            consumption = None
        if normalization not in [None, 'batch', 'switch']:
            normalization = 'batch'
        if output not in ['default', 'double', 'top', 'added']:
            output = 'default'
        if (residual_conv):
            residual = True
        if (skip_conv):
            skip_connections = True
        input_args = get_input_args(locals())

        layer_guide = self._calculate_layer_array(
            in_size, output_size, kernel_size, consumption)
        # if consumption

        layers = []
        for i, dil in enumerate(layer_guide):
            input_args['dilation'] = dil
            h = configurable_dilated_layer(**input_args)
            h.name = f"l{i}k{kernel_size} - Dil: {dil}"
            if debug:
                print(h.name)
            layers.append(h)
        self.layers = nn.ModuleList(layers)

        if normalization:
            norms = self._create_norm_array(
                in_channels, in_size, kernel_size, normalization, layer_guide)
            self.norms = nn.ModuleList(norms)

        self._calculate_forward(normalization, skip_connections)

    def _calculate_forward(self,
                           normalization, skip_connections):
        if normalization:
            self.f_norm = lambda i, x: self.norms[i](x)
        else:
            self.f_norm = lambda i, x: x
        if skip_connections:
            self.forward = self._skip_forward
        else:
            self.forward = self._simple_forward

#    def forward(self, x):
#        x, skips = self.layers[0](x)
#        x = self.norms[0](x)
#        for layer, norm in zip(self.hs[1:], self.norms[1:]):
#            # for layer in self.layers[1:]:
#            x, skip = layer(x)
#            x = norm(x)
#            skips += skip
#        return skips

    def _simple_forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.f_norm(i, x)
        return

    def _skip_forward(self, x):
        x, skips = self.layers[0](x)
        x = self.f_norm(0, x)
        for i, layer in enumerate(self.layers[1:]):
            x, skip = layer(x)
            x = self.f_norm(i+1, x)
            skips += skip
        return skips

    @staticmethod
    def _create_norm_array(in_channels, in_size, kernel_size, normalization, layer_array):
        remaining = in_size
        norm_array = []
        for dil in layer_array:
            remaining -= dil * (kernel_size - 1)
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
                layer_array.append(kernel_size ** i)

            if consumption == 'padded':
                self.pad = self._get_receptive_field(kernel_size, layers_num)\
                    - (self._get_receptive_field(kernel_size,
                                                 layers_num - 1) + (output_size - 1))
                break
            elif consumption == 'minimum':
                self.trim = in_size - \
                    (self._get_receptive_field(
                        kernel_size, layers_num) + (output_size - 1))
                break
            elif consumption == 'full':
                if layers_num == 0:
                    k = remaining_input - (output_size - 1)
                    if k == 1:
                        break
                    layer_array.append(k)

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
