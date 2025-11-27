
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import trunc_normal_
from collections import OrderedDict
import numpy as np



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # Clustering layers inspired by SwAV
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class TMEHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, use_bn=False):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        # x = nn.functional.normalize(x, dim=-1, p=2) # We do not need this normalization layer
        return x




###################################################################################################


class Conv2DBlock(nn.Module):
    """Conv2DBlock with convolution followed by GroupNorm, ReLU activation and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, GroupNorm, ReLU and dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
                bias=False,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)





class TMEHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, use_bn=False):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = []

        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())

        # Middle layers
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())

        # Final layer
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)  # Use weight, not weight_orig
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)




class MaskModel(nn.Module):
    """ViT-UNet Mask Model with GroupNorm (single decoder for all masks)."""
    
    def __init__(self, encoder, num_masks, encoder_dim=768, drop_rate=0.1):
        super().__init__()

        self.encoder = encoder
        self.num_masks = num_masks
        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        self.mask_decoder = self._create_upsampling_branch(num_masks)
        
        self._initialize_weights()

    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(enable)

    def _initialize_weights(self):
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.decoder0.apply(init_fn)
        self.decoder1.apply(init_fn)
        self.decoder2.apply(init_fn)
        self.decoder3.apply(init_fn)
        self.mask_decoder.apply(init_fn)

    def _create_upsampling_branch(self, num_classes: int) -> nn.Module:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        return nn.Sequential(
            OrderedDict([
                ("bottleneck_upsampler", bottleneck_upsampler),
                ("decoder3_upsampler", decoder3_upsampler),
                ("decoder2_upsampler", decoder2_upsampler),
                ("decoder1_upsampler", decoder1_upsampler),
                ("decoder0_header", decoder0_header),
            ])
        )

    def _forward_upsample(self, images, f1, f2, f3, f4):
        b4 = self.mask_decoder.bottleneck_upsampler(f4)
        b3 = self.decoder3(f3)
        b3 = self.mask_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        
        b2 = self.decoder2(f2)
        b2 = self.mask_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        
        b1 = self.decoder1(f1)
        b1 = self.mask_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        
        b0 = self.decoder0(images)
        output = self.mask_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        
        return output

    def forward(self, images):
        with torch.no_grad():
            features = self.encoder.get_intermediate_layers(images)
            f1, f2, f3, f4 = features

            num_patches = f1.shape[1] - (4 + 1)
            feature_size = int(np.sqrt(num_patches))

            f1 = f1[:, 5:, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size).contiguous()
            f2 = f2[:, 5:, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size).contiguous()
            f3 = f3[:, 5:, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size).contiguous()
            f4 = f4[:, 5:, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size).contiguous()

        logits = self._forward_upsample(images, f1, f2, f3, f4)
        masks = torch.softmax(logits, dim=1)

        return {"masks": masks}





class ReconstructorModel(nn.Module):
    def __init__(self, encoder, encoder_dim=768, drop_rate=0.1):
        super().__init__()

        self.encoder = encoder

        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate

        # Set dimensions based on encoder size
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            # Dimensions for ViT-Large and larger models
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        # Shared decoder blocks
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # Single decoder for all masks
        self.image_decoder = self.create_upsampling_branch(3)
        
        self.apply_spectral_norm_to_decoder(self.image_decoder)
        self.initialize_weights()

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing in the encoder"""
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(enable)
            print(f"Gradient checkpointing set to {enable} in MaskModel encoder")
        else:
            print("Warning: MaskModel encoder does not support gradient checkpointing")


    def apply_spectral_norm_to_decoder(self,decoder_module):
        """
        Applies spectral normalization to all convolutional and linear layers in a decoder module,
        excluding batch norm and activation functions.
        """
        for name, module in decoder_module.named_children():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                SpectralNorm.apply(module, 'weight', n_power_iterations=1, dim=0, eps=1e-12)
            elif isinstance(module, nn.Sequential):
                self.apply_spectral_norm_to_decoder(module)

    def initialize_weights(self):
        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Check if spectral norm has been applied (weight_orig exists)
                if hasattr(m, 'weight_orig'):
                    nn.init.kaiming_normal_(m.weight_orig, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Check if spectral norm has been applied
                if hasattr(m, 'weight_orig'):
                    nn.init.xavier_normal_(m.weight_orig)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Apply initialization to all modules (including Deconv2DBlock)
        self.apply(init_fn)

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlock(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=self.bottleneck_dim,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlock(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlock(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlock(64, 64, dropout=self.drop_rate),
            nn.Conv2d(
                in_channels=64,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
            ),

        )

        return nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

    def _forward_upsample(
        self,
        images: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        self.check_tensor_for_nan(images, "images in _forward_upsample")
        self.check_tensor_for_nan(f1, "f1 in _forward_upsample")
        self.check_tensor_for_nan(f2, "f2 in _forward_upsample")
        self.check_tensor_for_nan(f3, "f3 in _forward_upsample")
        self.check_tensor_for_nan(f4, "f4 in _forward_upsample")

        b4 = branch_decoder.bottleneck_upsampler(f4)
        self.check_tensor_for_nan(b4, "b4 after bottleneck_upsampler")

        b3 = self.decoder3(f3)
        self.check_tensor_for_nan(b3, "b3 after decoder3")

        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        self.check_tensor_for_nan(b3, "b3 after decoder3_upsampler")

        b2 = self.decoder2(f2)
        self.check_tensor_for_nan(b2, "b2 after decoder2")

        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        self.check_tensor_for_nan(b2, "b2 after decoder2_upsampler")

        b1 = self.decoder1(f1)
        self.check_tensor_for_nan(b1, "b1 after decoder1")

        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        self.check_tensor_for_nan(b1, "b1 after decoder1_upsampler")

        b0 = self.decoder0(images)
        self.check_tensor_for_nan(b0, "b0 after decoder0")

        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))
        self.check_tensor_for_nan(branch_output, "branch_output")

        return branch_output

    def check_weights_for_nan(self):
        for name, param in self.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                nan_indices = torch.nonzero(torch.isnan(param), as_tuple=False)
                raise ValueError(f"NaN detected in trainable weights: {name} at indices: {nan_indices.tolist()}")

    def check_tensor_for_nan(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            nan_indices = torch.nonzero(torch.isnan(tensor), as_tuple=False)
            raise ValueError(f"NaN detected in {tensor_name} at indices: {nan_indices.tolist()}")

    def forward(self, masks):
        self.check_tensor_for_nan(masks, "input images")
        self.check_weights_for_nan()

        with torch.no_grad():
            # Get features from encoder
            features = self.encoder.get_intermediate_layers(masks)
            f1, f2, f3, f4 = features

            # Reshape features
            num_patches = f1.shape[1] - (4 + 1)  # 4 register tokens plus cls token
            feature_size = int(np.sqrt(num_patches))

            f1 = f1[:, (4+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size).contiguous()
            f2 = f2[:, (4+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size).contiguous()
            f3 = f3[:, (4+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size).contiguous()
            f4 = f4[:, (4+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size).contiguous()



        # Output RGB image
        reconstructed = self._forward_upsample(masks, f1, f2, f3, f4, self.image_decoder)
        return torch.tanh(reconstructed) 




# ============================ CellViT with BatchNorms purely for augmentation purposes ==================================


class CellViT(nn.Module):
    """
    CellViT for nuclei/background segmentation augmentation.
    Simplified from instance segmentation version - single decoder, 2-channel output.
    Uses BatchNorm2d to match pre-trained checkpoint architecture.
    
    Returns:
        dict with "masks": [B, 2, H, W] where channel 0=nuclei, channel 1=background
    """
    def __init__(self, encoder, encoder_dim=768, drop_rate=0.1):
        super().__init__()
        
        self.encoder = encoder
        
        # Freeze encoder for augmentation use
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.embed_dim = encoder_dim
        self.drop_rate = drop_rate

        # Set dimensions based on encoder size
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 256
        elif self.embed_dim < 1024:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512
        else:
            self.skip_dim_11 = 768
            self.skip_dim_12 = 384
            self.bottleneck_dim = 768

        # Shared decoder layers (using Conv2DBlockBN and Deconv2DBlockBN with BatchNorm)
        self.decoder0 = nn.Sequential(
            Conv2DBlockBN(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlockBN(32, 64, 3, dropout=self.drop_rate),
        )
        self.decoder1 = nn.Sequential(
            Deconv2DBlockBN(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlockBN(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlockBN(self.skip_dim_12, 128, dropout=self.drop_rate),
        )
        self.decoder2 = nn.Sequential(
            Deconv2DBlockBN(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlockBN(self.skip_dim_11, 256, dropout=self.drop_rate),
        )
        self.decoder3 = nn.Sequential(
            Deconv2DBlockBN(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )

        # Single decoder for nuclei/background (2 channels)
        self.nuclei_binary_map_decoder = self.create_upsampling_branch(2)

        self.initialize_weights()

    def initialize_weights(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)
        
        # Initialize final layer with smaller weights
        final_conv = self.nuclei_binary_map_decoder.decoder0_header[-1]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.normal_(final_conv.weight, std=0.01)
            nn.init.constant_(final_conv.bias, 0)

    def create_upsampling_branch(self, num_classes: int) -> nn.Module:
        """Create upsampling branch for binary segmentation."""
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=self.embed_dim,
            out_channels=self.bottleneck_dim,
            kernel_size=2, stride=2, padding=0, output_padding=0,
        )
        decoder3_upsampler = nn.Sequential(
            Conv2DBlockBN(self.bottleneck_dim * 2, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlockBN(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            Conv2DBlockBN(self.bottleneck_dim, self.bottleneck_dim, dropout=self.drop_rate),
            nn.ConvTranspose2d(self.bottleneck_dim, 256, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder2_upsampler = nn.Sequential(
            Conv2DBlockBN(256 * 2, 256, dropout=self.drop_rate),
            Conv2DBlockBN(256, 256, dropout=self.drop_rate),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder1_upsampler = nn.Sequential(
            Conv2DBlockBN(128 * 2, 128, dropout=self.drop_rate),
            Conv2DBlockBN(128, 128, dropout=self.drop_rate),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        decoder0_header = nn.Sequential(
            Conv2DBlockBN(64 * 2, 64, dropout=self.drop_rate),
            Conv2DBlockBN(64, 64, dropout=self.drop_rate),
            nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0),
        )

        decoder = nn.Sequential(
            OrderedDict([
                ("bottleneck_upsampler", bottleneck_upsampler),
                ("decoder3_upsampler", decoder3_upsampler),
                ("decoder2_upsampler", decoder2_upsampler),
                ("decoder1_upsampler", decoder1_upsampler),
                ("decoder0_header", decoder0_header),
            ])
        )
        return decoder

    def _forward_upsample(
        self,
        images: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
        f4: torch.Tensor,
        branch_decoder: nn.Sequential,
    ) -> torch.Tensor:
        b4 = branch_decoder.bottleneck_upsampler(f4)
        b3 = self.decoder3(f3)
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        
        b2 = self.decoder2(f2)
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))
        
        b1 = self.decoder1(f1)
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))
        
        b0 = self.decoder0(images)
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        return branch_output

    def forward(self, images, magnification='40x'):
        out_dict = {}
        num_registers = 4
        features = self.encoder.get_intermediate_layers(images)
        f1, f2, f3, f4 = features
        
        # Determine feature map size dynamically
        num_patches = f1.shape[1] - (num_registers + 1)
        feature_size = int(np.sqrt(num_patches))

        # Reshape features to [B, C, H, W]
        f1 = f1[:, (num_registers+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size)
        f2 = f2[:, (num_registers+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size)
        f3 = f3[:, (num_registers+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size)
        f4 = f4[:, (num_registers+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size)
        
        # Binary nuclei/background segmentation
        mask_logits = self._forward_upsample(
            images, f1, f2, f3, f4, self.nuclei_binary_map_decoder
        )
        
        # Apply softmax to get probabilities
        out_dict["masks"] = F.log_softmax(mask_logits, dim=1).exp()

        return out_dict


# Helper blocks with BatchNorm (for CellViT checkpoint compatibility)
class Conv2DBlockBN(nn.Module):
    """Conv2D block with BatchNorm (for CellViT checkpoint compatibility)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlockBN(nn.Module):
    """Deconv2D block with BatchNorm (for CellViT checkpoint compatibility)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)





class ADIOSMaskModel(nn.Module):
    """
    ADIOS mask model - UNet with skip connections operating on RGB images.
    Based on: Shi et al., "Adversarial Masking for Self-Supervised Learning", ICML 2022
    """
    def __init__(self, num_masks=3, img_size=224, filter_start=32, norm='gn'):
        super().__init__()
        
        self.num_masks = num_masks
        self.img_size = img_size
        
        num_blocks = int(np.log2(img_size) - 1)
        self.num_blocks = num_blocks
        
        c = filter_start
        
        if norm == 'in':
            conv_block = ConvINReLU
        elif norm == 'gn':
            conv_block = ConvGNReLU
        else:
            conv_block = ConvReLU
        
        if num_blocks == 4:
            enc_in = [3, c, 2*c, 2*c]
            enc_out = [c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c]
            dec_out = [2*c, 2*c, c, c]
        elif num_blocks == 5:
            enc_in = [3, c, c, 2*c, 2*c]
            enc_out = [c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c]
        elif num_blocks == 6:
            enc_in = [3, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, c, c, c, c]
        elif num_blocks == 7:
            enc_in = [3, c, c, c, c, 2*c, 2*c]
            enc_out = [c, c, c, c, 2*c, 2*c, 2*c]
            dec_in = [4*c, 4*c, 4*c, 4*c, 2*c, 2*c, 2*c]
            dec_out = [2*c, 2*c, 2*c, c, c, c, c]
        else:
            raise ValueError(f"num_blocks={num_blocks} not supported. Use 4, 5, 6, or 7.")
        
        self.down = nn.ModuleList()
        for i, o in zip(enc_in, enc_out):
            self.down.append(conv_block(i, o, 3, 1, 1))
        
        self.up = nn.ModuleList()
        for i, o in zip(dec_in, dec_out):
            self.up.append(conv_block(i, o, 3, 1, 1))
        
        self.featuremap_size = img_size // (2 ** (num_blocks - 1))
        bottleneck_dim = 2 * c * self.featuremap_size * self.featuremap_size
        
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(dec_out[-1], num_masks, kernel_size=1)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        batch_size = images.size(0)
        
        x_down = [images]
        skip = []
        
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(act, scale_factor=0.5, mode='nearest')
            x_down.append(act)
        
        x_up = self.mlp(x_down[-1])
        x_up = x_up.view(batch_size, -1, self.featuremap_size, self.featuremap_size)
        
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode='nearest')
        
        logits = self.final_conv(x_up)
        masks = F.softmax(logits, dim=1)
        
        return {"masks": masks}


class ConvReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.ReLU(inplace=True)
        )


class ConvINReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0):
        super(ConvINReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.ReLU(inplace=True)
        )


class ConvGNReLU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, groups=8):
        super(ConvGNReLU, self).__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.ReLU(inplace=True)
        )