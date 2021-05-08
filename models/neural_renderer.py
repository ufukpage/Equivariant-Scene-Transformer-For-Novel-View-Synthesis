import torch
import torch.nn as nn
from misc.utils import pretty_print_layers_info, count_parameters
from models.submodels import ResNet2d, ResNet3d, Projection, InverseProjection
from models.rotation_layers import SphericalMask, Rotate3d

from transforms3d.conversions import rotation_matrix_source_to_target


class View(nn.Module):

    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        batch_size = input.size(0)
        return input.view(batch_size, *self.shape)


# from timesformer_pytorch import TimeSformer ## Ilk faktorize edilenlerden biri


class NeuralRenderer(nn.Module):
    """Implements a Neural Renderer with an implicit scene representation that
    allows both forward and inverse rendering.

    The forward pass from 3d scene to 2d image is (rendering):
    Scene representation (input) -> ResNet3d -> Projection -> ResNet2d ->
    Rendered image (output)

    The inverse pass from 2d image to 3d scene is (inverse rendering):
    Image (input) -> ResNet2d -> Inverse Projection -> ResNet3d -> Scene
    representation (output)

    Args:
        img_shape (tuple of ints): Shape of the image input to the model. Should
            be of the form (channels, height, width).
        channels_2d (tuple of ints): List of channels for 2D layers in inverse
            rendering model (image -> scene).
        strides_2d (tuple of ints): List of strides for 2D layers in inverse
            rendering model (image -> scene).
        channels_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        strides_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        num_channels_inv_projection (tuple of ints): Number of channels in each
            layer of inverse projection unit from 2D to 3D.
        num_channels_projection (tuple of ints): Number of channels in each
            layer of projection unit from 2D to 3D.
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used when rotating voxel grid.

    Notes:
        Given the inverse rendering channels and strides, the model will
        automatically build a forward renderer as the transpose of the inverse
        renderer.
    """
    def __init__(self, img_shape, channels_2d, strides_2d, channels_3d,
                 strides_3d, num_channels_inv_projection, num_channels_projection,
                 mode='bilinear'):
        super(NeuralRenderer, self).__init__()
        self.img_shape = img_shape
        self.channels_2d = channels_2d
        self.strides_2d = strides_2d
        self.channels_3d = channels_3d
        self.strides_3d = strides_3d
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        self.mode = mode

        # Initialize layers

        # Inverse pass (image -> scene)
        # First transform image into a 2D representation
        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d,
                                         strides_2d)

        # Perform inverse projection from 2D to 3D
        input_shape = self.inv_transform_2d.output_shape
        self.inv_projection = InverseProjection(input_shape, num_channels_inv_projection)

        # Transform 3D inverse projection into a scene representation
        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape,
                                         channels_3d, strides_3d)
        # Add rotation layer
        self.rotation_layer = Rotate3d(self.mode)

        # Forward pass (scene -> image)
        # Forward renderer is just transpose of inverse renderer, so flip order
        # of channels and strides
        # Transform scene representation to 3D features
        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.inv_transform_3d.output_shape,
                                     forward_channels_3d, forward_strides_3d)

        # Layer for projection of 3D representation to 2D representation
        self.projection = Projection(self.transform_3d.output_shape,
                                     num_channels_projection)

        # Transform 2D features to rendered image
        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape,
                                     forward_channels_2d, forward_strides_2d,
                                     final_conv_channels_2d)

        # Scene representation shape is output of inverse 3D transformation
        self.scene_shape = self.inv_transform_3d.output_shape
        # Add spherical mask before scene rotation
        self.spherical_mask = SphericalMask(self.scene_shape)

    def render(self, scene):
        """Renders a scene to an image.

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        """
        features_3d = self.transform_3d(scene)
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def inverse_render(self, img):
        """Maps an image to a (spherical) scene representation.

        Args:
            img (torch.Tensor): Shape (batch_size, channels, height, width).
        """
        # Transform image to 2D features
        features_2d = self.inv_transform_2d(img)
        # Perform inverse projection
        features_3d = self.inv_projection(features_2d)
        # Map 3D features to scene representation
        scene = self.inv_transform_3d(features_3d)
        # Ensure scene is spherical
        return self.spherical_mask(scene)

    def rotate(self, scene, rotation_matrix):
        """Rotates scene by rotation matrix.

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
                (batch_size, 3, 3).
        """
        return self.rotation_layer(scene, rotation_matrix)

    def rotate_source_to_target(self, scene, azimuth_source, elevation_source,
                                azimuth_target, elevation_target):
        """Assuming the scene is being observed by a camera at
        (azimuth_source, elevation_source), rotates scene so camera is observing
        it at (azimuth_target, elevation_target).

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source.
            elevation_source (torch.Tensor): Shape (batch_size,). Elevation of source.
            azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target.
            elevation_target (torch.Tensor): Shape (batch_size,). Elevation of target.
        """
        return self.rotation_layer.rotate_source_to_target(scene,
                                                           azimuth_source,
                                                           elevation_source,
                                                           azimuth_target,
                                                           elevation_target)

    def forward(self, batch):
        """Given a batch of images and poses, infers scene representations,
        rotates them into target poses and renders them into images.

        Args:
            batch (dict): A batch of images and poses as returned by
                misc.dataloaders.scene_render_dataloader.

        Notes:
            This *must* be a batch as returned by the scene render dataloader,
            i.e. the batch must be composed of pairs of images of the same
            scene. Specifically, the first time in the batch should be an image
            of scene A and the second item in the batch should be an image of
            scene A observed from a different pose. The third item should be an
            image of scene B and the fourth item should be an image scene B
            observed from a different pose (and so on).
        """
        # Slightly hacky way of extracting model device. Device on which
        # spherical is stored is the one where model is too
        device = self.spherical_mask.mask.device
        imgs = batch["img"].to(device)
        params = batch["render_params"]
        azimuth = params["azimuth"].to(device)
        elevation = params["elevation"].to(device)

        # Infer scenes from images
        scenes = self.inverse_render(imgs)

        # Rotate scenes so that for every pair of rendered images, the 1st
        # one will be reconstructed as the 2nd and then 2nd will be
        # reconstructed as the 1st
        swapped_idx = get_swapped_indices(azimuth.shape[0])

        # Each pair of indices in the azimuth vector corresponds to the same
        # scene at two different angles. Therefore performing a pairwise swap,
        # the first index will correspond to the second index in the original
        # vector. Since we want to rotate camera angle 1 to camera angle 2 and
        # vice versa, we can use these swapped angles to define a target
        # position for the camera
        azimuth_swapped = azimuth[swapped_idx]
        elevation_swapped = elevation[swapped_idx]
        scenes_swapped = \
            self.rotate_source_to_target(scenes, azimuth, elevation,
                                         azimuth_swapped, elevation_swapped)

        # Swap scenes, so rotated scenes match with original inferred scene.
        # Specifically, we have images x1, x2 from which we inferred the scenes
        # z1, z2. We then rotated these scenes into z1' and z2'. Now z1' should
        # be almost equal to z2 and z2' should be almost equal to z1, so we swap
        # the order of z1', z2' to z2', z1' so we can easily render them to
        # x1 and x2.
        scenes_rotated = scenes_swapped[swapped_idx]

        # Render scene using model
        rendered = self.render(scenes_rotated)

        return imgs, rendered, scenes, scenes_rotated

    def print_model_info(self):
        """Prints detailed information about model, such as how input shape is
        transformed to output shape and how many parameters are trained in each
        block.
        """
        print("Forward renderer")
        print("----------------\n")
        pretty_print_layers_info(self.transform_3d, "3D Layers")
        print("\n")
        pretty_print_layers_info(self.projection, "Projection")
        print("\n")
        pretty_print_layers_info(self.transform_2d, "2D Layers")
        print("\n")

        print("Inverse renderer")
        print("----------------\n")
        pretty_print_layers_info(self.inv_transform_2d, "Inverse 2D Layers")
        print("\n")
        pretty_print_layers_info(self.inv_projection, "Inverse Projection")
        print("\n")
        pretty_print_layers_info(self.inv_transform_3d, "Inverse 3D Layers")
        print("\n")

        print("Scene Representation:")
        print("\tShape: {}".format(self.scene_shape))
        # Size of scene representation corresponds to non zero entries of
        # spherical mask
        print("\tSize: {}\n".format(int(self.spherical_mask.mask.sum().item())))

        print("Number of parameters: {}\n".format(count_parameters(self)))

    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return {
            "img_shape": self.img_shape,
            "channels_2d": self.channels_2d,
            "strides_2d": self.strides_2d,
            "channels_3d": self.channels_3d,
            "strides_3d": self.strides_3d,
            "num_channels_inv_projection": self.num_channels_inv_projection,
            "num_channels_projection": self.num_channels_projection,
            "mode": self.mode
        }

    def save(self, filename):
        """Saves model and its config.

        Args:
            filename (string): Path where model will be saved. Should end with
                '.pt' or '.pth'.
        """
        torch.save({
            "config": self.get_model_config(),
            "state_dict": self.state_dict()
        }, filename)

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = NeuralRenderer(
            img_shape=config["img_shape"],
            channels_2d=config["channels_2d"],
            strides_2d=config["strides_2d"],
            channels_3d=config["channels_3d"],
            strides_3d=config["strides_3d"],
            num_channels_inv_projection=config["num_channels_inv_projection"],
            num_channels_projection=config["num_channels_projection"],
            mode=config["mode"]
        )
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model

# from models.transformers import ViTransformer2DEncoder, ViTransformer3DEncoder
from models.vision_transformers import ViTransformer2DEncoder, ViTransformer3DEncoder, ViTransformer2DEncoderWrapper, \
    ViTransformer3DEncoderWrapper, ViewEmbedding, FixedEmbedding
from x_transformers import Encoder
import  vit_pytorch
from nystrom_attention import Nystromer
from einops.layers.torch import Rearrange


class TransformerRendererV0(NeuralRenderer):

    def __init__(self, config):
        super(TransformerRendererV0, self).__init__(img_shape=config["img_shape"], channels_2d=config["channels_2d"],
                                                    strides_2d=config["strides_2d"],channels_3d=config["channels_3d"],
                                                    strides_3d=config["strides_3d"],
                                                    num_channels_inv_projection=config["num_channels_inv_projection"],
                                                    num_channels_projection=config["num_channels_projection"],
                                                    mode=config["mode"])
        self.config = config

        output_size = config["img_shape"][1] // config["patch_sizes"][0]

        self.inv_transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][2],  # config["patch_size_3d"],
            pos_emd=True,
            transformer=Nystromer(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
            ),
            dim=output_size * 2
        )
        self.inv_transform_3d = nn.Sequential(
            self.inv_transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size * 2)
        )

        self.transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            pos_emd=False,
            transformer=Nystromer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
            ),
            dim=output_size
        )

        self.transform_3d = nn.Sequential(
            self.transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size)
        )

    def print_model_info(self):
        print("Number of parameters: {}\n".format(count_parameters(self)))

    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return self.config

    @staticmethod
    def load_model(filename):
        model_dict = torch.load(filename, map_location="cpu")
        model = TransformerRendererV0(model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])
        return model


class TransformerRendererV01(TransformerRendererV0):

    def __init__(self, config):
        super(TransformerRendererV01, self).__init__(config)

        output_size = config["img_shape"][1] // config["patch_sizes"][0]

        self.transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            pos_emd=False,
            view_emd=True,
            transformer=Nystromer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
            ),
            dim=output_size
        )

        self.transform_3d = nn.Sequential(
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size)
        )

    def render_angles(self, scene, az, el):
        features_3d = self.transform_3d(self.transformer_3d(scene,  az, el))
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def forward(self, batch):
        device = self.spherical_mask.mask.device
        imgs = batch["img"].to(device)
        params = batch["render_params"]
        azimuth = params["azimuth"].to(device)
        elevation = params["elevation"].to(device)

        # Infer scenes from images
        scenes = self.inverse_render(imgs)

        swapped_idx = get_swapped_indices(azimuth.shape[0])

        azimuth_swapped = azimuth[swapped_idx]
        elevation_swapped = elevation[swapped_idx]
        scenes_swapped = \
            self.rotate_source_to_target(scenes, azimuth, elevation,
                                         azimuth_swapped, elevation_swapped)

        scenes_rotated = scenes_swapped[swapped_idx]

        rendered = self.render_angles(scenes_rotated, azimuth_swapped, elevation_swapped)

        return imgs, rendered, scenes, scenes_rotated


from models.timesformer import  DepthFormer


class DepthFormerRenderer(TransformerRendererV0):

    def __init__(self, config):
        super(DepthFormerRenderer, self).__init__(config)
        self.config = config

        output_size = config["img_shape"][1] // config["patch_sizes"][0]

        self.inv_transformer_3d = DepthFormer(
            volume_depth=output_size,
            image_size=output_size,
            patch_size=config["patch_sizes"][2],  # config["patch_size_3d"],
            dim=output_size * 2,
            heads=1,
            dim_head=output_size,
            channels=output_size
        )
        self.inv_transform_3d = nn.Sequential(
            self.inv_transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size * 2)
        )

        self.transformer_3d = DepthFormer(
            volume_depth=output_size,
            image_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            dim_head=output_size,
            heads=1,
            dim=output_size
        )

        self.transform_3d = nn.Sequential(
            self.transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size)
        )

    @staticmethod
    def load_model(filename):
        model_dict = torch.load(filename, map_location="cpu")
        model = DepthFormerRenderer(model_dict["config"])
        model.load_state_dict(model_dict["state_dict"])
        return model


class TransformerRenderer(NeuralRenderer):
    def __init__(self, config):
        super(TransformerRenderer, self).__init__(img_shape=config["img_shape"], channels_2d=config["channels_2d"],
                                                  strides_2d=config["strides_2d"],channels_3d=config["channels_3d"],
                                                  strides_3d=config["strides_3d"],
                                                  num_channels_inv_projection=config["num_channels_inv_projection"],
                                                  num_channels_projection=config["num_channels_projection"],
                                                  mode=config["mode"])
        self.config = config
        self.inv_transformer_2d = ViTransformer2DEncoder(
            image_size=config["img_shape"][1],
            patch_size=config["patch_sizes"][0],
            transformer=Nystromer(
                dim=128,
                depth=1,
                heads=config["heads"][0],
                num_landmarks=256,
            ),
            dim=128
        )
        output_size = config["img_shape"][1]//config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
                                self.inv_transformer_2d,
                                Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                )

        self.inv_projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][1],
            channels=128,
            dim=1024,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][1]
            )
        )

        self.inv_projection = nn.Sequential(
                                self.inv_projection_transformer,
                                Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
                                # View([32, 32, 32, 32])
                            )

        self.inv_transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][2], # config["patch_size_3d"],
            transformer=Nystromer(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
                num_landmarks=256,
            ),
            dim=output_size * 2
        )
        self.inv_transform_3d = nn.Sequential(
                                self.inv_transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size*2)
                            )

        self.spherical_mask = SphericalMask((output_size*2, output_size, output_size, output_size))

        self.transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            transformer=Nystromer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
                num_landmarks=256,
            ),
            dim=output_size
        )

        self.transform_3d = nn.Sequential(
                                self.transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size)
                            )

        self.projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][4],
            channels=output_size * output_size,
            transformer=Nystromer(
                dim=256,
                depth=1,
                heads=config["heads"][4],
                num_landmarks=256,
            ),
            dim=256
        )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )

        self.transformer_2d = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][5],
            channels=256,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][5]
            ),
            dim=1024,
            # use_embeddings=False
        )

        self.transform_2d = nn.Sequential(
                                        self.transformer_2d,
                                        # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
                                        View([4, config["img_shape"][1], config["img_shape"][2]]),
                                        nn.Conv2d(4, 3, 1)
                                        )

    def print_model_info(self):
        print("Number of parameters: {}\n".format(count_parameters(self)))

    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return self.config

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = TransformerRenderer(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


class TransformerRendererV3(TransformerRenderer):
    def __init__(self, config):
        super(TransformerRendererV3, self).__init__(config)
        self.config = config
        self.inv_transformer_2d = ViTransformer2DEncoder(
            image_size=config["img_shape"][1],
            patch_size=config["patch_sizes"][0],
            transformer=Nystromer(
                dim=128,
                depth=1,
                heads=config["heads"][0],
                num_landmarks=256,
            ),
            dim=128
        )
        output_size = config["img_shape"][1]//config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
                                self.inv_transformer_2d,
                                Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                )

        self.inv_projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][1],
            channels=128,
            dim=1024,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][1]
            )
        )

        self.inv_projection = nn.Sequential(
                                self.inv_projection_transformer,
                                Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
                                # View([32, 32, 32, 32])
                            )

        self.inv_transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][2], # config["patch_size_3d"],
            transformer=Nystromer(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
                num_landmarks=256,
            ),
            dim=output_size * 2,
            multi_dim_pos_embedding=True
        )
        self.inv_transform_3d = nn.Sequential(
                                self.inv_transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size*2)
                            )

        self.spherical_mask = SphericalMask((output_size*2, output_size, output_size, output_size))

        self.transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            transformer=Nystromer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
                num_landmarks=256,
            ),
            dim=output_size,
            use_pos_embedding=False
        )

        self.transform_3d = nn.Sequential(
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size)
                            )

        self.projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][4],
            channels=output_size * output_size,
            transformer=Nystromer(
                dim=256,
                depth=1,
                heads=config["heads"][4],
                num_landmarks=256,
            ),
            dim=256
        )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )

        self.transformer_2d = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][5],
            channels=256,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][5]
            ),
            dim=1024
        )

        self.transform_2d = nn.Sequential(
                                        self.transformer_2d,
                                        # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
                                        View([4, config["img_shape"][1], config["img_shape"][2]]),
                                        nn.Conv2d(4, 3, 1)
                                        )

    def rotate_pos_emd(self, azimuth_source, elevation_source,
                             azimuth_target, elevation_target):

        rot_matrix = rotation_matrix_source_to_target(azimuth_source, elevation_source,
                                         azimuth_target, elevation_target)

        pos_emd = self.inv_transformer_3d.get_pos_embedding(cat=True).unsqueeze(0)
        return pos_emd #torch.matmul(rot_matrix, torch.cat((pos_emd, pos_emd), dim=0))

    def render(self, scene):
        features_3d = self.transformer_3d(scene, self.inv_transformer_3d.get_pos_embedding())
        features_3d = self.transform_3d(features_3d)

        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def forward(self, batch):

        device = self.spherical_mask.mask.device
        imgs = batch["img"].to(device)
        params = batch["render_params"]
        azimuth = params["azimuth"].to(device)
        elevation = params["elevation"].to(device)

        # Infer scenes from images
        scenes = self.inverse_render(imgs)

        swapped_idx = get_swapped_indices(azimuth.shape[0])

        azimuth_swapped = azimuth[swapped_idx]
        elevation_swapped = elevation[swapped_idx]
        scenes_swapped = \
            self.rotate_source_to_target(scenes, azimuth, elevation,
                                         azimuth_swapped, elevation_swapped)

        # rotated_pos_emd = self.rotate_pos_emd(azimuth, elevation, azimuth_swapped, elevation_swapped)
        # self.rotated_pos_emd = rotated_pos_emd[swapped_idx]

        scenes_rotated = scenes_swapped[swapped_idx]

        # Render scene using model
        rendered = self.render(scenes_rotated)

        return imgs, rendered, scenes, scenes_rotated

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = TransformerRendererV3(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


class VanillaTransformerRenderer(TransformerRenderer):

    def __init__(self, config):
        super(VanillaTransformerRenderer, self).__init__(config)

        self.inv_transformer_2d.transformer = Encoder(
                dim=128,
                depth=1,
                heads=config["heads"][0],
                rel_pos_bias=True
            )

        output_size = config["img_shape"][1]//config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
                                self.inv_transformer_2d,
                                Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                )

        self.inv_projection_transformer.transformer = Encoder(
                dim=1024,
                depth=1,
                heads=config["heads"][1],
                rel_pos_bias=True
            )

        self.inv_projection = nn.Sequential(
                                self.inv_projection_transformer,
                                Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
                                # View([32, 32, 32, 32])
                            )

        self.inv_transformer_3d.transformer = Encoder(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
                rel_pos_bias=True
            )

        self.inv_transform_3d = nn.Sequential(
                                self.inv_transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size*2)
                            )

        self.transformer_3d.transformer = Encoder(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
                rel_pos_bias=True
            )

        self.transform_3d = nn.Sequential(
                                self.transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size)
                            )

        self.projection_transformer.transformer = Encoder(
                dim=256,
                depth=1,
                heads=config["heads"][4],
                rel_pos_bias=True
            )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )

        self.transformer_2d.transformer = Encoder(
                dim=1024,
                depth=1,
                heads=config["heads"][4],
                rel_pos_bias=True
            )

        self.transform_2d = nn.Sequential(
                                        self.transformer_2d,
                                        # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
                                        View([4, config["img_shape"][1], config["img_shape"][2]]),
                                        nn.Conv2d(4, 3, 1)
                                        )

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = VanillaTransformerRenderer(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


class TransformerRendererNop(TransformerRenderer):

    def __init__(self, config):
        super(TransformerRendererNop, self).__init__(config)

        self.inv_transformer_2d = ViTransformer2DEncoder(
            image_size=config["img_shape"][1],
            patch_size=config["patch_sizes"][0],
            pos_emd=True,
            transformer=Nystromer(
                dim=128,
                depth=1,
                heads=config["heads"][0],
            ),
            dim=128
        )
        output_size = config["img_shape"][1] // config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
            self.inv_transformer_2d,
            Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
        )

        self.inv_projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][1],
            channels=128,
            dim=1024,
            pos_emd=False,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][1]
            )
        )

        self.inv_projection = nn.Sequential(
            self.inv_projection_transformer,
            Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
            # View([32, 32, 32, 32])
        )

        self.inv_transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][2],  # config["patch_size_3d"],
            pos_emd=False,
            transformer=Nystromer(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
            ),
            dim=output_size * 2
        )
        self.inv_transform_3d = nn.Sequential(
            self.inv_transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size * 2)
        )

        self.transformer_3d = ViTransformer3DEncoder(
            volume_size=output_size,
            patch_size=config["patch_sizes"][3],
            channels=output_size * 2,
            pos_emd=False,
            transformer=Nystromer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
            ),
            dim=output_size
        )

        self.transform_3d = nn.Sequential(
            self.transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size)
        )

        self.projection_transformer = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][4],
            channels=output_size * output_size,
            pos_emd=False,
            transformer=Nystromer(
                dim=256,
                depth=1,
                heads=config["heads"][4]
            ),
            dim=256
        )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )

        self.transformer_2d = ViTransformer2DEncoder(
            image_size=output_size,
            patch_size=config["patch_sizes"][5],
            channels=256,
            pos_emd=False,
            transformer=Nystromer(
                dim=1024,
                depth=1,
                heads=config["heads"][5]
            ),
            dim=1024,
            # use_embeddings=False
        )

        self.transform_2d = nn.Sequential(
            self.transformer_2d,
            # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
            View([4, config["img_shape"][1], config["img_shape"][2]]),
            nn.Conv2d(4, 3, 1)
        )

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = TransformerRendererNop(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


from linformer import Linformer


class LinformerRenderer(TransformerRenderer):
    def __init__(self, config):
        super(LinformerRenderer, self).__init__(config)
        num_patches = (config["img_shape"][1] // config["patch_sizes"][0]) ** 2
        self.inv_transformer_2d.transformer = Linformer(
                dim=128,
                heads=config["heads"][0],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )

        output_size = config["img_shape"][1] // config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
            self.inv_transformer_2d,
            Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
        )
        num_patches = (output_size // config["patch_sizes"][1]) ** 2
        self.inv_projection_transformer.transformer = Linformer(
                dim=1024,
                heads=config["heads"][1],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )

        self.inv_projection = nn.Sequential(
            self.inv_projection_transformer,
            Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
        )
        num_patches = (output_size // config["patch_sizes"][2]) ** 3
        self.inv_transformer_3d.transformer=Linformer(
                dim=output_size * 2,
                heads=config["heads"][2],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )

        self.inv_transform_3d = nn.Sequential(
            self.inv_transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size * 2)
        )
        num_patches = (output_size // config["patch_sizes"][3]) ** 3
        self.transformer_3d.transformer = Linformer(
                dim=output_size,
                heads=config["heads"][3],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )
        self.transform_3d = nn.Sequential(
            self.transformer_3d,
            Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                      c=output_size)
        )

        num_patches = (output_size // config["patch_sizes"][4]) ** 2
        self.projection_transformer.transformer = Linformer(
                dim=256,
                heads=config["heads"][4],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )
        num_patches = (output_size // config["patch_sizes"][5]) ** 2
        self.transformer_2d.transformer=Linformer(
                dim=1024,
                heads=config["heads"][5],
                k=128,
                one_kv_head=True,
                share_kv=True,
                seq_len=num_patches,
                depth=1
            )

        self.transform_2d = nn.Sequential(
            self.transformer_2d,
            # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
            View([4, config["img_shape"][1], config["img_shape"][2]]),
            nn.Conv2d(4, 3, 1)
        )

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = LinformerRenderer(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


from linear_attention_transformer import LinearAttentionTransformer


class LinearTransformerRenderer(TransformerRenderer):
    def __init__(self, config):
        super(LinearTransformerRenderer, self).__init__(config)
        num_patches = (config["img_shape"][1] // config["patch_sizes"][0]) ** 2
        self.inv_transformer_2d.transformer = LinearAttentionTransformer(
                dim=128,
                depth=1,
                heads=config["heads"][0],
                n_local_attn_heads=config["heads"][0],
                max_seq_len=num_patches,
                local_attn_window_size=32
            )
        output_size = config["img_shape"][1]//config["patch_sizes"][0]
        self.inv_transform_2d = nn.Sequential(
                                self.inv_transformer_2d,
                                Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                )
        num_patches = (output_size // config["patch_sizes"][1]) ** 2
        self.inv_projection_transformer.transformer = LinearAttentionTransformer(
                dim=1024,
                depth=1,
                heads=config["heads"][1],
                n_local_attn_heads=config["heads"][1],
                max_seq_len=num_patches,
                local_attn_window_size=32

        )

        self.inv_projection = nn.Sequential(
                                self.inv_projection_transformer,
                                Rearrange('b (p p2) (d p1) -> b d p1 p p2', p1=output_size, p2=output_size)
                                # View([32, 32, 32, 32])
                            )
        num_patches = (output_size // config["patch_sizes"][2]) ** 3
        self.inv_transformer_3d.transformer = LinearAttentionTransformer(
                dim=output_size * 2,
                depth=1,
                heads=config["heads"][2],
                n_local_attn_heads=config["heads"][2],
                max_seq_len=num_patches,
                local_attn_window_size=32
            )

        self.inv_transform_3d = nn.Sequential(
                                self.inv_transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size*2)
                            )

        num_patches = (output_size // config["patch_sizes"][3]) ** 3

        self.transformer_3d.transformer=LinearAttentionTransformer(
                dim=output_size,
                depth=1,
                heads=config["heads"][3],
                n_local_attn_heads=config["heads"][3],
                max_seq_len=num_patches,
                local_attn_window_size=32
            )

        self.transform_3d = nn.Sequential(
                                self.transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size)
                            )

        num_patches = (output_size // config["patch_sizes"][4]) ** 2
        self.projection_transformer.transformer=LinearAttentionTransformer(
                dim=256,
                depth=1,
                heads=config["heads"][4],
                n_local_attn_heads=config["heads"][4],
                max_seq_len=num_patches,
                local_attn_window_size=32
            )

        self.projection = nn.Sequential(Rearrange('b c d h w -> b (c d) h w'),
                                        self.projection_transformer,
                                        Rearrange('b (p1 p2) c -> b c p1 p2', p1=output_size, p2=output_size)
                                        )

        num_patches = (output_size // config["patch_sizes"][5]) ** 2
        self.transformer_2d.transformer = LinearAttentionTransformer(
                dim=1024,
                depth=1,
                heads=config["heads"][5],
                n_local_attn_heads=config["heads"][5],
                max_seq_len=num_patches,
                local_attn_window_size=32
            )

        self.transform_2d = nn.Sequential(
                                        self.transformer_2d,
                                        # Rearrange('b (p1 p2) c -> b c p1 p2', p1=config["img_shape"][1], p2=config["img_shape"][2]),
                                        View([4, config["img_shape"][1], config["img_shape"][2]]),
                                        nn.Conv2d(4, 3, 1)
                                        )
    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = LinearTransformerRenderer(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model

from models.vivit import DeViT
class TransformerRendererV2(TransformerRenderer):
    def __init__(self, config):
        super(TransformerRendererV2, self).__init__(config)

        output_size = config["img_shape"][1]//config["patch_sizes"][0]

        self.inv_transformer_3d = DeViT(volume_size=output_size, patch_size=config["patch_sizes"][2],
                                        depth_size=output_size, depth=1, in_channels=output_size, dim_head=output_size,
                                        heads=config["heads"][2], dim=output_size * 2)

        self.inv_transform_3d = nn.Sequential(
                                self.inv_transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size*2)
                            )

        self.transformer_3d = DeViT(volume_size=output_size, patch_size=config["patch_sizes"][3],
                                    depth_size=output_size, depth=1, in_channels=output_size*2, dim_head=output_size,
                                    heads=config["heads"][3], dim=output_size)

        self.transform_3d = nn.Sequential(
                                self.transformer_3d,
                                Rearrange('b (h w d) c -> b c h w d', h=output_size, w=output_size, d=output_size,
                                          c=output_size)
                            )

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = TransformerRendererV2(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


class SimpleTransformerRenderer(NeuralRenderer):
    def __init__(self, config):
        super(SimpleTransformerRenderer, self).__init__(img_shape=config["img_shape"], channels_2d=config["channels_2d"],
                                                  strides_2d=config["strides_2d"],channels_3d=config["channels_3d"],
                                                  strides_3d=config["strides_3d"],
                                                  num_channels_inv_projection=config["num_channels_inv_projection"],
                                                  num_channels_projection=config["num_channels_projection"],
                                                  mode=config["mode"])
        self.config = config
        self.inv_transform_2d = ViTransformer2DEncoder(
            image_size=config["img_shape"][1],
            patch_size=config["patch_size_2d"],
            transformer=Encoder(
                dim=1024,
                depth=1,
                heads=1,
                ff_glu=True,
                rel_pos_bias=True,
                use_scalenorm=True
            )
        )

        self.rotation_layer = Rotate3d(self.mode)
        self.inv_transform_3d = ViTransformer3DEncoder(
            volume_size=32,
            patch_size=config["patch_size_3d"],
            transformer=Encoder(
                dim=2048,
                depth=1,
                heads=1,
                ff_glu=True,
                rel_pos_bias=True,
                use_scalenorm=True
            )
        )

        self.transform_3d = ViTransformer3DEncoder(
            volume_size=32,
            patch_size=config["patch_size_3d"],
            transformer=Encoder(
                dim=2048,
                depth=1,
                heads=1,
                ff_glu=True,
                rel_pos_bias=True,
                use_scalenorm=True
            )
        )

        self.transform_2d = ViTransformer2DEncoder(
            image_size=32,
            patch_size=8,
            channels=32*32,
            transformer=Encoder(
                dim=1024,
                depth=1,
                heads=1,
                ff_glu=True,
                rel_pos_bias=True,
                use_scalenorm=True
            )
        )
        # uplift3d = nn.Linear(1024, 1024)
        # self.uplift3d = nn.Conv2d(256, 1024, kernel_size=1)
        self.final_render = nn.Conv2d(1, 3, kernel_size=1)
        self.spherical_mask = SphericalMask((32, 32, 32, 32))

    def print_model_info(self):
        pass

    def inverse_render(self, img):
        batch_size = img.shape[0]
        feats_1d = self.inv_transform_2d(img)  # (1, 1000)

        feats_2d = feats_1d.view(batch_size, feats_1d.shape[1], 32, -1)
        # feats_2d = self.uplift3d(feats_2d)
        uplifted_feats = feats_2d.view(batch_size, feats_2d.shape[2], 32, 32, -1)

        feats_3d = self.inv_transform_3d(uplifted_feats)
        scene = feats_3d.view(batch_size, 32, 32, 32, -1)
        return self.spherical_mask(scene)

    def render(self, scene):
        batch_size = scene.shape[0]
        features_3d = self.transform_3d(scene)
        features_3d = features_3d.view(batch_size, 32, 32, 32, -1)
        batch_size, channels, depth, height, width = features_3d.shape
        # Reshape 3D -> 2D
        features_2d = features_3d.view(batch_size, channels * depth, height, width)
        # features_2d = self.transform_2d(features_2d)
        features_2d = self.transform_2d(features_2d)
        features_2d = features_2d.view(batch_size, 1, 128, 128)
        features_2d = self.final_render(features_2d)
        return torch.sigmoid(features_2d)

    @staticmethod
    def load_model(filename):
        """Loads a NeuralRenderer model from saved model config and weights.

        Args:
            filename (string): Path where model was saved.
        """
        model_dict = torch.load(filename, map_location="cpu")
        config = model_dict["config"]
        # Initialize a model based on config
        model = SimpleTransformerRenderer(config)
        # Load weights into model
        model.load_state_dict(model_dict["state_dict"])
        return model


def get_swapped_indices(length):
    """Returns a list of swapped index pairs. For example, if length = 6, then
    function returns [1, 0, 3, 2, 5, 4], i.e. every index pair is swapped.

    Args:
        length (int): Length of swapped indices.
    """
    return [i + 1 if i % 2 == 0 else i - 1 for i in range(length)]
