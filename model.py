import importlib

import torch.nn as nn

from buildingblocks import SingleConv, DoubleConv, ExtResNetBlock, create_encoders, \
    create_decoders
from utils import number_of_features_per_level


class Abstract3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid, basic_module, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, testing=False,
                 conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(in_channels, 
                                        f_maps, 
                                        basic_module, 
                                        conv_kernel_size, 
                                        conv_padding, 
                                        layer_order,
                                        num_groups, 
                                        pool_kernel_size)

        # create decoder path
        self.decoders = create_decoders(f_maps, 
                                        basic_module, 
                                        conv_kernel_size, 
                                        conv_padding, 
                                        layer_order, 
                                        num_groups,
                                        upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs
        # logits and it's up to the user to normalize it before visualising with tensorboard or computing validation metric
        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        super(UNet3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=SingleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_padding=conv_padding,
                                     **kwargs)


class ResidualUNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             final_sigmoid=final_sigmoid,
                                             basic_module=ExtResNetBlock,
                                             f_maps=f_maps,
                                             layer_order=layer_order,
                                             num_groups=num_groups,
                                             num_levels=num_levels,
                                             is_segmentation=is_segmentation,
                                             conv_padding=conv_padding,
                                             **kwargs)


class UNet2D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, **kwargs):
        if conv_padding == 1:
            conv_padding = (0, 1, 1)
        super(UNet2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     final_sigmoid=final_sigmoid,
                                     basic_module=DoubleConv,
                                     f_maps=f_maps,
                                     layer_order=layer_order,
                                     num_groups=num_groups,
                                     num_levels=num_levels,
                                     is_segmentation=is_segmentation,
                                     conv_kernel_size=(1, 3, 3),
                                     pool_kernel_size=(1, 2, 2),
                                     conv_padding=conv_padding,
                                     **kwargs)


def get_model(model_config):
    def _model_class(class_name):
        modules = ['pytorch3dunet.unet3d.model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
