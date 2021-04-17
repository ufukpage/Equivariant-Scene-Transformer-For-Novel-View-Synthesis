import sys
import torch
import json
from misc.dataloaders import scene_render_dataset
from misc.quantitative_evaluation import get_dataset_psnr
from models.neural_renderer import load_model
from models.neural_renderer import NeuralRenderer, TransformerRenderer, SimpleTransformerRenderer, TransformerRendererV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get path to experiment folder from command line arguments
if len(sys.argv) != 4:
    raise(RuntimeError("Wrong arguments, use python experiments_psnr.py <model_path> <dataset_folder> <model_type>"))

model_path = sys.argv[1]
data_dir = sys.argv[2]  # This is usually one of "chairs-test" and "cars-test"
model_type = sys.argv[3]

if model_type == "neq":
    model = NeuralRenderer.load_model(model_path)
elif model_type == "t":
    # Set up renderer
    model = TransformerRenderer.load_model(model_path)
elif model_type == "tv2":
    model = TransformerRendererV2.load_model(model_path)
else:
    # Set up renderer
    model = SimpleTransformerRenderer.load_model(model_path)

model = model.to(device)

# Initialize dataset
dataset = scene_render_dataset(path_to_data=data_dir, img_size=(3, 128, 128),
                               crop_size=128, allow_odd_num_imgs=True, get_img_data=True)

# Calculate PSNR
with torch.no_grad():
    psnrs = get_dataset_psnr(device, model, dataset, data_dir, source_img_idx_shift=64,
                             batch_size=2, max_num_scenes=None)
