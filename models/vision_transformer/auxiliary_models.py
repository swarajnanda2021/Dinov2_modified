
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


class SpectralNorm:
    """
    Spectral Normalization module that constrains the spectral norm 
    of the weight matrix to be 1. This helps stabilize training by 
    preventing exploding gradients and maintaining Lipschitz continuity.
    """
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module, do_power_iteration=True):
        weight = getattr(module, self.name + '_orig')
        weight_mat = weight

        if weight_mat.dim() > 2:
            weight_mat = weight_mat.reshape(weight_mat.shape[0], -1)

        height = weight_mat.size(0)
        width = weight_mat.size(1)
        
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Normalize in correct dimensions
                    v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps)
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma

        # Reshape back if needed
        if weight.dim() != weight_mat.dim():
            weight = weight.view_as(getattr(module, self.name + '_orig'))

        return weight

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)

        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        v = F.normalize(weight.new_empty(weight.size(0)).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", nn.Parameter(weight.data))
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


class TMEHead_SpectralNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, bottleneck_dim=256, nlayers=3, use_bn=False):
        super().__init__()
        nlayers = max(nlayers, 1)
        layers = []
        
        # First layer
        linear = nn.Linear(in_dim, hidden_dim)
        SpectralNorm.apply(linear, 'weight', n_power_iterations=1, dim=0, eps=1e-12)
        layers.append(linear)
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Middle layers
        for _ in range(nlayers - 2):
            linear = nn.Linear(hidden_dim, hidden_dim)
            SpectralNorm.apply(linear, 'weight', n_power_iterations=1, dim=0, eps=1e-12)
            layers.append(linear)
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        # Final layer
        linear = nn.Linear(hidden_dim, bottleneck_dim)
        SpectralNorm.apply(linear, 'weight', n_power_iterations=1, dim=0, eps=1e-12)
        layers.append(linear)
        
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight_orig, std=.02)  # Note: we use weight_orig now
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)




class MaskModel_SpectralNorm(nn.Module):
    def __init__(self, encoder, num_masks, encoder_dim=768, drop_rate=0.1):
        super().__init__()
        
        self.encoder = encoder
        self.num_masks = num_masks
        
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
        # self.mask_decoder = self.create_upsampling_branch(num_masks)
        # Separate decoders for each mask
        self.mask_decoders = nn.ModuleList([
            self.create_upsampling_branch(1) for _ in range(num_masks)
        ])
        
        self.initialize_weights()

        # Apply spectral norm to all decoder layers
        self.apply_spectral_norm_to_decoder(self.decoder0)
        self.apply_spectral_norm_to_decoder(self.decoder1)
        self.apply_spectral_norm_to_decoder(self.decoder2)
        self.apply_spectral_norm_to_decoder(self.decoder3)
        for decoder in self.mask_decoders:
            self.apply_spectral_norm_to_decoder(decoder)

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

    def single_decoder_initialize_weights(self):
        def init_fn(m):
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

        # Apply initialization to all modules
        self.apply(init_fn)
        
        # For single decoder head
        # Special initialization for Deconv2DBlock
        for module in self.modules():
            if isinstance(module, Deconv2DBlock):
                for layer in module.block:
                    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                            
    def initialize_weights(self):
        # First initialize shared components
        def init_fn(m):
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

        # Initialize shared decoders
        self.decoder0.apply(init_fn)
        self.decoder1.apply(init_fn)
        self.decoder2.apply(init_fn)
        self.decoder3.apply(init_fn)

        # Initialize each mask decoder with different random seeds and scales
        for idx, decoder in enumerate(self.mask_decoders):
            # Set a different random seed for each decoder
            torch.manual_seed(42 + idx)  # Different seed for each decoder
            
            for module in decoder.modules():
                if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    # Scale the initialization variance differently for each decoder
                    scale = 1.0 + (idx * 0.1)  # Each decoder gets progressively larger init
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu', a=scale)
                    if module.bias is not None:
                        # Add small offset to biases based on decoder index
                        nn.init.constant_(module.bias, idx * 0.01)
                elif isinstance(module, nn.BatchNorm2d):
                    # Slightly different starting points for BatchNorm
                    nn.init.constant_(module.weight, 1.0 + (idx * 0.05))
                    nn.init.constant_(module.bias, idx * 0.01)

                # Special handling for Deconv2DBlock
                if isinstance(module, Deconv2DBlock):
                    for layer in module.block:
                        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                            scale = 1.0 + (idx * 0.1)
                            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu', a=scale)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, idx * 0.01)

        # Reset RNG state after initialization
        torch.manual_seed(torch.initial_seed())

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

    def forward(self, images):
        self.check_tensor_for_nan(images, "input images")
        self.check_weights_for_nan()

        with torch.no_grad():
            # Get features from encoder
            features = self.encoder.get_intermediate_layers(images)
            f1, f2, f3, f4 = features
            
            # Reshape features
            num_patches = f1.shape[1] - (4 + 1)  # 4 register tokens plus cls token
            feature_size = int(np.sqrt(num_patches))
            
            f1 = f1[:, (4+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size).contiguous()
            f2 = f2[:, (4+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size).contiguous()
            f3 = f3[:, (4+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size).contiguous()
            f4 = f4[:, (4+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size).contiguous()
        
        
        
        # Pre-allocate output tensor
        batch_size = images.size(0)
        H, W = images.size(2), images.size(3)
        all_masks = torch.empty(batch_size, self.num_masks, H, W,
                               device=images.device, dtype=images.dtype)

        # Process each mask separately
        mask_outputs = []
        for decoder in self.mask_decoders:
            logits = self._forward_upsample(images, f1, f2, f3, f4, decoder)
            mask_outputs.append(logits)
            
        # Concatenate all mask outputs and apply softmax
        masks = torch.cat(mask_outputs, dim=1)  # [B, N_masks, H, W]
        masks = torch.softmax(masks, dim=1)
        
        return {
            "masks": masks,  # [B, N_masks, H, W]
        }


class Conv2DBlock(nn.Module): # with batch norms
    """Conv2DBlock with convolution followed by batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for convolution
        out_channels (int): Number of output channels for convolution
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            #nn.BatchNorm2d(out_channels),# avoiding batch norms like the plague since I will be using these for adversarial training.
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Deconv2DBlock(nn.Module):
    """Deconvolution block with ConvTranspose2d followed by Conv2d, batch-normalisation, ReLU activation and dropout

    Args:
        in_channels (int): Number of input channels for deconv block
        out_channels (int): Number of output channels for deconv and convolution.
        kernel_size (int, optional): Kernel size for convolution. Defaults to 3.
        dropout (float, optional): Dropout. Defaults to 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
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
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            #nn.BatchNorm2d(out_channels),# avoiding batch norms like the plague if using things in an adversarial setup
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True),
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
    def __init__(self, encoder, num_masks, encoder_dim=768, drop_rate=0.1):
        super().__init__()

        self.encoder = encoder
        self.num_masks = num_masks

        #for parameter in self.encoder.parameters():
        #    parameter.requires_grad = False

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
        self.mask_decoder = self.create_upsampling_branch(num_masks)
        

        self.initialize_weights()

    def set_grad_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing in the encoder"""
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(enable)
            print(f"Gradient checkpointing set to {enable} in MaskModel encoder")
        else:
            print("Warning: MaskModel encoder does not support gradient checkpointing")


    def initialize_weights(self):
        def init_fn(m):
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

    def forward(self, images):
        self.check_tensor_for_nan(images, "input images")
        self.check_weights_for_nan()

        with torch.no_grad():
            # Get features from encoder
            features = self.encoder.get_intermediate_layers(images)
            f1, f2, f3, f4 = features

            # Reshape features
            num_patches = f1.shape[1] - (4 + 1)  # 4 register tokens plus cls token
            feature_size = int(np.sqrt(num_patches))

            f1 = f1[:, (4+1):, :].permute(0, 2, 1).view(f1.shape[0], -1, feature_size, feature_size).contiguous()
            f2 = f2[:, (4+1):, :].permute(0, 2, 1).view(f2.shape[0], -1, feature_size, feature_size).contiguous()
            f3 = f3[:, (4+1):, :].permute(0, 2, 1).view(f3.shape[0], -1, feature_size, feature_size).contiguous()
            f4 = f4[:, (4+1):, :].permute(0, 2, 1).view(f4.shape[0], -1, feature_size, feature_size).contiguous()


        # Single forward pass
        logits = self._forward_upsample(images, f1, f2, f3, f4, self.mask_decoder)
        masks = torch.softmax(logits, dim=1)

        
        return {
            "masks": masks,  # [B, N_masks, H, W]
        }







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

