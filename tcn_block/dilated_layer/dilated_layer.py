from tcn_block.misc import get_input_args
import torch.nn as nn


class configurable_dilated_layer(nn.Module):
    def __init__(self,
                 in_channels, kernel_size, output_size,
                 dilation=1,
                 padding=0, groups=1, bias=True,
                 residual=False, residual_conv=False,
                 skip_connections=False, skip_conv=False,
                 debug=False, *args, **kwargs):
        super().__init__()

        input_args = get_input_args(locals())

        self.output_size = output_size

        self.dilated_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                      padding=padding,
                                      dilation=dilation, groups=groups, bias=bias)

        if residual_conv:
            self.conv_res = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                      kernel_size=1, padding=0, dilation=1,
                                      groups=1, bias=bias)

            self.residual_output = self._residual_conv_out
        else:
            self.residual_output = lambda out, res: out + res

        if skip_conv:
            self.conv_skip = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                       kernel_size=1, padding=0, dilation=1,
                                       groups=1, bias=bias)
            self.skip_output = self._skip_conv_forward
        else:
            self.skip_output = self._skip_forward

        self.forward = self._select_forward(**input_args)

        if debug:
            print(
                f"Created dilated {'R' if residual else ''}{'S' if skip_connections else ''} layer, K[{kernel_size}]")

    def _select_forward(self,
                        residual,
                        skip_connections,
                        *args, **kwargs):

        self.selected_single_forward = self._residual_forward if residual else self._simple_forward

        if skip_connections:
            return self.skip_output
        else:
            return self.selected_single_forward

    def _simple_forward(self, x):
        return self.dilated_conv(x)

    def _residual_forward(self, x):
        out = self.dilated_conv(x)

        x = x.narrow(-1, 0, out.shape[-1])
        residual_out = self.residual_output(out, x)
        return residual_out

    def _residual_conv_out(self, out, res):
        return self.conv_res(out + res)

    def _skip_forward(self, x):
        out = self.selected_single_forward(x)
        skip = out.narrow(-1, 0, self.output_size)
        return (out, skip)

    def _skip_conv_forward(self, x):
        out = self.selected_single_forward(x)
        skip = self.conv_skip(out.narrow(-1, 0, self.output_size))
        return (out, skip)
