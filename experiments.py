import json
import os
import sys
import time
import torch
from misc.dataloaders import scene_render_dataloader
from models.neural_renderer import NeuralRenderer, TransformerRenderer, SimpleTransformerRenderer, TransformerRendererV2\
    , LinformerRenderer, LinearTransformerRenderer, VanillaTransformerRenderer, TransformerRendererV3, \
    TransformerRendererNop, TransformerRendererV0,TransformerRendererV01, DepthFormerRenderer
from models.vision_transformers import ViTransformer2DEncoder, ViTransformer3DEncoder
from training.training import Trainer

if "__main__" == __name__:
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Get path to data from command line arguments
    if len(sys.argv) < 2:
        raise(RuntimeError("Wrong arguments, use python experiments.py <config>"))
    path_to_config = sys.argv[1]

    load_path = sys.argv[2] if len(sys.argv) >= 3 else None # config.json ./2021-04-01_23-32_chairs-experiment/model.pt 2021-04-12_00-20_chairs-experiment

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
    elif config["model_name"] == "tv0":
        model = TransformerRendererV0(
            config
        )
    elif config["model_name"] == "df":
        model = DepthFormerRenderer(
            config
        )
    elif config["model_name"] == "tv01":
        model = TransformerRendererV01(
            config
        )
    elif config["model_name"] == "vt":
        model = VanillaTransformerRenderer(
            config
        )
    elif config["model_name"] == "t":

        model = TransformerRenderer(
            config
        )
    elif config["model_name"] == "tv2":
        model = TransformerRendererV2(
            config
        )
    elif config["model_name"] == "tv3":
        model = TransformerRendererV3(
            config
        )
    elif config["model_name"] == "l":
        model = LinformerRenderer(config)
    elif config["model_name"] == "lt":
        model = LinearTransformerRenderer(config)
    elif config["model_name"] == "nop":

        model = TransformerRendererNop(
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
                      ssim_loss_weight=config["ssim_loss_weight"], feature_loss=config["feature_loss"]
                      , iteration_verbose=config["iteration_verbose"] if "iteration_verbose" in config else 1)

    dataloader = scene_render_dataloader(path_to_data=config["path_to_data"],
                                         batch_size=config["batch_size"],
                                         img_size=config["img_shape"],
                                         crop_size=128,
                                         get_img_data=True)

    # Optionally set up test_dataloader
    if config["path_to_test_data"]:
        test_dataloader = scene_render_dataloader(path_to_data=config["path_to_test_data"],
                                                  batch_size=config["batch_size"],
                                                  img_size=config["img_shape"],
                                                  crop_size=128, get_img_data=True)
    else:
        test_dataloader = None

    print("PID: {}".format(os.getpid()))

    # Train renderer, save generated images, losses and model
    trainer.train(dataloader, config["epochs"], save_dir=directory,
                  save_freq=config["save_freq"], test_dataloader=test_dataloader, load_path=load_path,
                  resume_epoch=config["resume_epoch"])

    # Print best losses
    print("Model id: {}".format(config["id"]))
    print("Best train loss: {:.4f}".format(min(trainer.epoch_loss_history["total"])))
    print("Best validation loss: {:.4f}".format(min(trainer.val_loss_history["total"])))
