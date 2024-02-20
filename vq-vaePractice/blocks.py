import torch

from torch import nn
from torch.nn import functional as F

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        '''
        The block for residual Stack.
        
        Arguments:
            num_hiddens(int): number of input channels.
            num_residual_layers(int): depth of residual_layers.
            num_residual_hiddens(int): number or output chahnnels.

        Inputs:
            [B, num_hiddens, H, W].
        
        Ouputs:
            [B, num_residual_hiddens, H, W].
        '''
        super(ResidualStack, self).__init__()
        # See Section 4.1 of "Neural Discrete Representation Learning".
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    # maintain the size of input image.
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.ReLU(),
                    # maintain the size of input image.
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        kernel_size=1,
                    ),
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = h + layer(h)
        # ResNet V1-style.
        return torch.relu(h)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_downsampling_layers, num_residual_layers, num_residual_hiddens):
        '''
        The block for Encoding. 

        Arguments:
            in_channels (int): number of in_channels.
            num_hiddens (int): number of out_channels. (ResidualStack's input_channels).
            num_downsampling_layers (int): depth of encoding layers.
            num_residual_layers (int): depth of residual_layers. (ResidualStack's depth).
            num_residual_hiddens (int): num or final out_channles. (ResidualStack's output_channels).
        
        Inputs:
            [B, in_channels, H, W].

        Outputs:
            [B, num_residual_hiddens, H/2^num_downsampling_layers, W/2^downsampling_layers].
        '''
        super(Encoder, self).__init__()
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)

            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                # resize to 1/2 the size of input image.
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())
        
        conv.add_module(
            "final_conv",
            # maintain the size of input image.
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1,
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
    
    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_upsampling_layers, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        '''
        The block for Decoding.

        Arguments:
            embedding_dim (int): number of input channals.
            num_hiddens (int): number of out_channels. (ResidualStack's input_channels).
            num_upsampling_layers (int): depth of decoder layers.
            num_residual_layers (int): depth of residual_layers. (ResidualStack's depth).
            num_residual_hiddens (int): num or final out_channles. (ResidualStack's output_channels).
        
        Inputs:
            [B, num_residual_hiddens, H/2^num_upsampling_layers, W/2^upsampling_layers].

        Ouputs:
            [B, embedding_dim, H, W]

        '''
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)

            else:
                (in_channels, out_channels) = (num_hiddens // 2, 3)

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon