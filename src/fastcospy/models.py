"""
FastCOSPY a U-Net+LSTM+Attention Model
--------------------------------------

This module implements a U-Net-based model with:
- Down and Up blocks with convolutions and optional dropout.
- A spatial attention gate at the bottleneck.
- Latent representation processed by an LSTM.
- Multi-energy output predictions per spatial location.

Created for fast emulation of the private COSPY framework.

All classes are designed to be imported and used programmatically. No scripts or example runs
are included here — see the 'examples/' directory for usage demonstrations.
"""

__all__ = ["DoubleConvolution", "Down", "Up", "SpatialAttentionGate", "UNetLSTMAttention"]

import torch
import torch.nn as nn
import torch.nn.functional as F

GN_GROUPS = 8

class DoubleConvolution(nn.Module):

    """
    A two-layer convolutional block with optional dropout and GroupNorm.

    This block performs two sequential Conv2d -> GroupNorm -> ReLU operations, with an optional
    dropout at the end.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : float, optional (default=0.)
        Dropout probability. If 0., no dropout is applied.
    groups : int, optional (default=GN_GROUPS)
        Number of groups for GroupNorm.
    """

    def __init__(self, in_channels, out_channels, dropout=0., groups=GN_GROUPS):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0.:
            layers.append(nn.Dropout2d(p=dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W), where B is the batch size.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H, W).
        """

        return self.block(x)
    

class Down(nn.Module):

    """
    The Down block of the U-Net architecture with optional dropout.
    This block performs a MaxPool2d -> DoubleConvolution.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : float, optional (default=0.)
        Dropout probability. If 0., no dropout is applied.
    """

    def __init__(self, in_channels, out_channels, dropout=0.):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.my_double = DoubleConvolution(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Forward pass through the Down block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W), where B is the batch size.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, H / 2, W / 2).
        """

        x = self.pool(x)
        x = self.my_double(x)
        
        return x
    

class Up(nn.Module):
    
    """
    The Up block of the U-Net architecture with optional bilinear interpolation.
    This block performs an Upsample -> DoubleConvolution if bilinear is True or
    ConvTranspose2d -> DoubleConvolution if bilinear is False.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bilinear : bool, optional (default=True)
        Choose between upsampling or convtranspose2d.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvolution(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvolution(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the Up block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, H, W), where B is the batch size and where C_in is
            the number of channels of the upsampled input (typically half of in_channels by
            construction in this U-Net).

        skip : torch.Tensor
            Skip connection tensor of shape (B, skip_channels, H * 2, W * 2), where skip_channels
            is the number of channels of the skip connection (typically half of in_channels by
            construction in this U-Net)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, out_channels, 2 * H,  2 * W).
        """
        x = self.up(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]) # Ensures
        # that the size of x matches the size of skip.

        x = torch.cat([skip, x], dim=1)
        
        x = self.conv(x)
        return x
    

class SpatialAttentionGate(nn.Module):

    """
    Spatial attention gate used in U-Net architectures.

    This module applies attention on a feature map 'x' using a gating signal 'g'.
    It computes attention coefficients that highlight relevant spatial regions 
    in 'x' based on the context provided by 'g'.

    The block performs:
    1. 1x1 convolutions on 'x' and 'g' to reduce channels to 'F_int'.
    2. Upsampling of the gating signal to match spatial dimensions of 'x'.
    3. ReLU activation followed by a 1x1 convolution and sigmoid to compute attention weights.
    4. Element-wise multiplication of attention weights with the input 'x'.

    Parameters
    ----------
    F_g : int
        Number of channels in the gating signal 'g'.
    F_l : int
        Number of channels in the input feature map 'x'.
    F_int : int
        Number of intermediate channels used inside the attention block.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.GroupNorm(GN_GROUPS, F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.GroupNorm(GN_GROUPS, F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the spatial attention gate.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (B, F_l, H, W), where B is batch size, 
            F_l is the number of channels, and, H and W are spatial dimensions.
        g : torch.Tensor
            Gating signal of shape (B, F_g, H, W), where F_g is the number of channels,
            and, H and W are spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, F_l, H, W), where the input 'x' has been
            modulated by spatial attention weights derived from 'g'.
        """
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)

        return x * alpha
    

class UNetLSTMAttention(nn.Module):

    """
    U-Net with LSTM and spatial attention for multi-energy prediction.

    This module implements a U-Net encoder-decoder architecture with:
    - Downsampling and upsampling blocks,
    - A spatial attention gate at the bottleneck,
    - A latent representation processed with an LSTM,
    - Linear layers to produce multi-energy outputs.

    The model is designed to predict 'n_energies' outputs per spatial location,
    making it suitable for applications where each pixel requires multiple correlated predictions.

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_energies : int, optional, default=7
        Number of energy outputs to predict per spatial location.
    latent_channels : int, optional, default=128
        Number of channels in the latent representation after the last upsampling block.
    lstm_hidden_size : int, optional, default=128
        Hidden size of the LSTM used to process the latent representation.
    """

    def __init__(self, n_channels, n_energies=7, latent_channels=128, lstm_hidden_size=128):
        super().__init__()
        C1, C2, C3, C4, C5 = 96, 192, 384, 768, 1536

        self.inc   = DoubleConvolution(n_channels, C1)
        self.down1 = Down(C1, C2)
        self.down2 = Down(C2, C3)
        self.down3 = Down(C3, C4)
        self.down4 = Down(C4, C4, dropout=0.2)

        self.up1 = Up(C5, C3)
        self.up2 = Up(C4, C2)
        self.up3 = Up(C3, C1)
        self.up4 = Up(C2, latent_channels)

        self.lstm = nn.LSTM(input_size=latent_channels, hidden_size=lstm_hidden_size, num_layers=1, batch_first=True)

        self.energy_head = nn.Linear(lstm_hidden_size, 1)

        self.n_energies = n_energies

        self.att_bottleneck = SpatialAttentionGate(F_g=C4, F_l=C4, F_int=C3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the UNetLSTMAttention model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, n_channels, H, W), where B is the batch size,
            n_channels is the number of input channels, and H, W are spatial dimensions.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, n_energies, H, W), containing the predicted values
            for each energy channel at each spatial location.
        """

        B, _, H, W = x.shape

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.att_bottleneck(x5, x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        C = x.shape[1]

        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B * H * W, 1, C)
        x = x.repeat(1, self.n_energies, 1)

        lstm_out, _ = self.lstm(x)
        lstm_out = self.energy_head(lstm_out)

        out = lstm_out.view(B, H, W, self.n_energies)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out