import json
import os
import sys
import time
import torch
from misc.dataloaders import scene_render_dataloader
from models.neural_renderer import NeuralRenderer, TransformerRenderer, SimpleTransformerRenderer
from models.vision_transformers import ViTransformer2DEncoder, ViTransformer3DEncoder
from training.training import Trainer

if "__main__" == __name__:
    test_transformer = 0
    if test_transformer:
        model_dict = torch.load("./2021-03-22_21-42_chairs-experiment/model.pt", map_location="cpu")
        config = model_dict["config"]
        # print(model_dict["state_dict"])
        for param_tensor in model_dict["state_dict"]:
            print(param_tensor, "\t", model_dict["state_dict"][param_tensor].size())
        exit(1)
        from nystrom_attention import Nystromformer

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path_to_config = sys.argv[1]
        with open(path_to_config) as file:
            config = json.load(file)

        from x_transformers import ViTransformerWrapper, TransformerWrapper, Encoder, Decoder
        import torch.nn as nn
        from nystrom_attention import Nystromer
        inv_transform_2d = ViTransformer2DEncoder(
            image_size=config["img_shape"][1],
            patch_size=config["patch_size_2d"],
            transformer=Nystromformer(
                dim=1024,
                depth=6,
                heads=8,
                num_landmarks=256,
            ),
            dim=1024
        ).to(device)

        inv_transform_3d = ViTransformer3DEncoder(
            volume_size=32,
            patch_size=config["patch_size_3d"],
            transformer=Encoder(
                dim=2048,
                depth=6,
                heads=8,
                ff_glu=True
            ),
            dim=2048
        ).to(device)

        # uplift3d = nn.Linear(1024, 1024)
        uplift3d = nn.Conv2d(256, 1024, kernel_size=1).to(device)

        # encoder.forward = def
        img = torch.randn(3, 3, 128, 128).to(device)
        feats_1d = inv_transform_2d(img)  # (1, 1000)

        # inv_projection isminde olacak
        feats_2d = feats_1d.view(3, feats_1d.shape[1], 32, -1)
        uplifted_feats = uplift3d(feats_2d)
        feats_3d = uplifted_feats.view(3, uplifted_feats.shape[2], 32, 32, -1)

        feats_3d = inv_transform_3d(feats_3d)
        scene = feats_3d.view(3, 32, 32, 32, -1)
        exit(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Get path to data from command line arguments
    if len(sys.argv) < 2:
        raise(RuntimeError("Wrong arguments, use python experiments.py <config>"))
    path_to_config = sys.argv[1]

    load_path = sys.argv[2] if len(sys.argv) >= 3 else None # config.json ./2021-04-01_23-32_chairs-experiment/model.pt

    # Open config file
    with open(path_to_config) as file:
        config = json.load(file)

    # Set up directory to store experiments
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in directory
    with open(directory + '/config.json', 'w') as file:
        json.dump(config, file)

    # Set up renderer
    if config["model_name"] == "neq":
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
    elif config["model_name"] == "t":
        # Set up renderer
        model = TransformerRenderer(
            config
        )
    else:
        # Set up renderer
        model = SimpleTransformerRenderer(
            config
        )

    model.print_model_info()

    if load_path:
        # from models.neural_renderer import load_model
        print("load path:", load_path)
        model = model.load_model(load_path)
    model = model.to(device)

    if config["multi_gpu"]:
        model = torch.nn.DataParallel(model)

    # Set up trainer for renderer
    trainer = Trainer(device, model, lr=config["lr"],
                      rendering_loss_type=config["loss_type"],
                      ssim_loss_weight=config["ssim_loss_weight"], feature_loss=config["feature_loss"])

    dataloader = scene_render_dataloader(path_to_data=config["path_to_data"],
                                         batch_size=config["batch_size"],
                                         img_size=config["img_shape"],
                                         crop_size=128)

    # Optionally set up test_dataloader
    if config["path_to_test_data"]:
        test_dataloader = scene_render_dataloader(path_to_data=config["path_to_test_data"],
                                                  batch_size=config["batch_size"],
                                                  img_size=config["img_shape"],
                                                  crop_size=128)
    else:
        test_dataloader = None

    print("PID: {}".format(os.getpid()))

    # Train renderer, save generated images, losses and model
    trainer.train(dataloader, config["epochs"], save_dir=directory,
                  save_freq=config["save_freq"], test_dataloader=test_dataloader)

    # Print best losses
    print("Model id: {}".format(config["id"]))
    print("Best train loss: {:.4f}".format(min(trainer.epoch_loss_history["total"])))
    print("Best validation loss: {:.4f}".format(min(trainer.val_loss_history["total"])))
