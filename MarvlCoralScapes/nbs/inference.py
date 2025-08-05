import torch 
import numpy as np 
import albumentations as A

from torchmetrics.classification import JaccardIndex
import numpy as np 


import numpy as np
from PIL import Image

from matplotlib import pyplot as plt 
from coralscapesScripts.visualization import color_label, color_correctness

from coralscapesScripts.datasets.preprocess import preprocess_inference
from coralscapesScripts.segmentation.model import predict

import cv2
from tqdm import tqdm

from torch.utils.data import DataLoader
from coralscapesScripts.datasets.dataset import Coralscapes
from coralscapesScripts.datasets.preprocess import get_preprocessor
from coralscapesScripts.datasets.utils import calculate_weights

from coralscapesScripts.segmentation.model import Benchmark_Run
from coralscapesScripts.segmentation.benchmarking import launch_benchmark

from coralscapesScripts.visualization import show_samples

from coralscapesScripts.logger import Logger, save_benchmark_run
from coralscapesScripts.io import setup_config, get_parser, update_config_with_args
import copy
from coralscapesScripts.segmentation.eval import Evaluator 

import argparse
parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--model_checkpoint', type=str, default=None, help='Path to model checkpoint')
parser_arg.add_argument('--is_coralscape', action='store_true', help='Set this flag if evaluating on Coralscapes dataset')
args_arg = parser_arg.parse_args()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

id_to_color_map = lambda x: Coralscapes.train_id_to_color[x]

device_count = torch.cuda.device_count()
for i in range(device_count):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device

# cfg = setup_config(config_path='../configs/deeplabv3+resnet50.yaml', config_base_path='../configs/base.yaml')
cfg = setup_config(config_path='configs/segformer-mit-b5.yaml', config_base_path='configs/base.yaml')
# cfg = setup_config(config_path='../configs/dpt-dinov2-base_lora.yaml', config_base_path='../configs/base.yaml')

args_input = "--batch-size=1 --batch-size-eval=1"
args_input = args_input.split(" ")

parser = get_parser()
args = parser.parse_args(args_input)

cfg = update_config_with_args(cfg, args)
cfg

transforms = {}
for split in cfg.augmentation:
    transforms[split] = A.Compose(
        [
            getattr(A, transform_name)(**transform_params) for transform_name, transform_params
                                                                 in cfg.augmentation[split].items()
        ]
    )

# train_dataset = Coralscapes(root = cfg.data.root, split = 'train', transform = transforms["train"])
transform_target = cfg.training.eval.transform_target if cfg.training.eval is not None and cfg.training.eval.transform_target is not None else True
# val_dataset = Coralscapes(root = cfg.data.root, split = 'val', transform = transforms["val"], transform_target=transform_target) 
test_dataset = Coralscapes(root = cfg.data.root, split = 'test', transform = transforms["test"], transform_target=transform_target)

# train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=4, drop_last=True)
# val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=4)

print(
    # len(train_loader), 
    # len(val_loader), 
    len(test_loader)
)

# cfg.model.checkpoint = "/home/yuxiang/data/murky_coral_dataset/Model_checkpoints/segformer-b5-finetuned-coralscapes-1024-1024_20"

# cfg.model.checkpoint = "/home/yuxiang/data/coralscapes_1080p/final_model_checkpoints/segformer_mit_b5_more_aug_epoch100"
# is_coralscape = True

# # cfg.model.checkpoint = "/home/yuxiang/Implementation/MarvlCoralScapes/model_checkpoints/fine_tune_notebook_lr0006/model_2025051320" # old
# cfg.model.checkpoint = "/home/yuxiang/Implementation/MarvlCoralScapes/model_checkpoints/murky_dataset_lr0024_weighted/model_2025053003"
# cfg.model.checkpoint = "/home/yuxiang/data/murky_coral_dataset/Model_checkpoints/segformer-b5-finetuned-coralscapes-1024-1024_20" # shruti's old model

if args_arg.model_checkpoint is not None:
    cfg.model.checkpoint = args_arg.model_checkpoint
is_coralscape = args_arg.is_coralscape

print("train_dataset.N_classes:-", test_dataset.N_classes)
print("cfg.model.checkpoint:--", cfg.model.checkpoint)

benchmark_run = Benchmark_Run(run_name = cfg.run_name, model_name = cfg.model.name, 
                                    N_classes = test_dataset.N_classes, device= device, 
                                    model_kwargs = cfg.model.kwargs,
                                    model_checkpoint = cfg.model.checkpoint,
                                    lora_kwargs = cfg.lora,
                                    training_hyperparameters = cfg.training,
                                    is_coralscape = is_coralscape)

benchmark_run.print_trainable_parameters()

metric_dict = {
                "iou": JaccardIndex(task="multiclass", num_classes=int(benchmark_run.N_classes), ignore_index = 0, average='none').to(device)
                }

if hasattr(benchmark_run, "preprocessor"):
    evaluator = Evaluator(metric_dict = metric_dict, N_classes = benchmark_run.N_classes, device = benchmark_run.device, preprocessor = benchmark_run.preprocessor, eval_params=benchmark_run.eval)
else:
    evaluator = Evaluator(metric_dict = metric_dict, N_classes = benchmark_run.N_classes, device = benchmark_run.device, eval_params=benchmark_run.eval)

print("Replaced with the following model:", cfg.model.checkpoint)

test_metric_results = evaluator.evaluate_model(test_loader, benchmark_run.model, split = "test", is_coralscape = is_coralscape)
print(test_metric_results)

with open(f"./metrics.txt", "a") as f:
    f.write(str(test_metric_results) + "\n")
    test_metric_results["iou"] = {test_dataset.id2label[class_id]:class_iou for class_id, class_iou in enumerate(test_metric_results["iou"])}
    f.write(str(np.array(list(test_metric_results["iou"].values()))[1:].mean()) + "\n")



# # To save images uncomment this code chunk
# from torchmetrics.classification import JaccardIndex
# import numpy as np 


# import numpy as np
# from PIL import Image

# from matplotlib import pyplot as plt 
# from coralscapesScripts.visualization import color_label, color_correctness

# from coralscapesScripts.datasets.preprocess import preprocess_inference
# from coralscapesScripts.segmentation.model import predict

# import cv2
# from tqdm import tqdm

# for idx in tqdm(range(len(test_dataset))):
#     inference_image_file_path = test_dataset.images[idx]
#     inference_image = np.array(Image.open(inference_image_file_path).convert('RGB'))

#     inference_image_target = test_dataset.targets[idx]
#     inference_target = np.array(Image.open(inference_image_target))
#     inference_target_colors = color_label(inference_target)

#     metric_dict = {
#                 "iou": JaccardIndex(task="multiclass", num_classes=int(benchmark_run.N_classes), ignore_index = 0, average='none').to(device)
#                 }
    
    # preprocessed_batch, window_dims = preprocess_inference(inference_image, transforms["test"], benchmark_run)

    # benchmark_run.model.eval()
    # with torch.no_grad():
    #     inference_prediction = predict(preprocessed_batch, 
    #                                     benchmark_run,
    #                                     window_dims = window_dims,
    #                                     is_coralscape = is_coralscape)
        
    # inference_prediction_colors = color_label(inference_prediction)
    # cv2.imwrite("<output inference path>"+"/output_"+str(idx)+".png", inference_prediction_colors)

