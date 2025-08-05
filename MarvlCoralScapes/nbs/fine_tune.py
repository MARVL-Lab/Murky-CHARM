import gc
import torch 
import numpy as np

import albumentations as A

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


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


device_count = torch.cuda.device_count()
for i in range(device_count):
    print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device

cfg = setup_config(config_path='configs/segformer-mit-b5.yaml', config_base_path='configs/base.yaml')

args_input = "--run-name=murky_dataset_lr0024_weighted --batch-size=4 --batch-size-eval=4 --epochs=20 --lr=0.024"
args_input = args_input.split(" ")

parser = get_parser()
args = parser.parse_args(args_input)

cfg = update_config_with_args(cfg, args)
cfg_logger = copy.deepcopy(cfg)

transforms = {}
for split in cfg.augmentation:
    transforms[split] = A.Compose(
        [
            getattr(A, transform_name)(**transform_params) for transform_name, transform_params
                                                                 in cfg.augmentation[split].items()
        ]
    )

print(cfg.data.root)
train_dataset = Coralscapes(root = cfg.data.root, split = 'train', transform = transforms["train"])

transform_target = cfg.training.eval.transform_target if cfg.training.eval is not None and cfg.training.eval.transform_target is not None else True

val_dataset = Coralscapes(root = cfg.data.root, split = 'val', transform = transforms["val"], transform_target=transform_target) 
test_dataset = Coralscapes(root = cfg.data.root, split = 'test', transform = transforms["test"], transform_target=transform_target)


train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size_eval, shuffle=False, num_workers=1)

print(len(train_loader), len(val_loader), len(test_loader))

weight = calculate_weights(train_dataset).to(device) if(cfg.data.weight) else None
if(cfg.data.weight):
    cfg.training.loss.weight = weight

print("train_dataset.N_classes:- ", train_dataset.N_classes)
print("epochs:-", args.epochs)
print("lr:-", args.lr)

benchmark_run = Benchmark_Run(run_name = cfg.run_name, model_name = cfg.model.name, 
                                    N_classes = train_dataset.N_classes, device= device, 
                                    model_kwargs = cfg.model.kwargs,
                                    model_checkpoint = cfg.model.checkpoint,
                                    lora_kwargs = cfg.lora,
                                    training_hyperparameters = cfg.training)


# Freeze all parameters
for param in benchmark_run.model.parameters():
    param.requires_grad = False

for param in benchmark_run.model.decode_head.parameters():
    param.requires_grad = True

logger = Logger(
    project = cfg.logger.wandb_project,
    benchmark_run = benchmark_run,
    log_epochs = cfg.logger.log_epochs,
    log_checkpoint=2,  # Save every 2 epochs
    config = cfg_logger,
    checkpoint_dir = "./"
)

benchmark_run.print_trainable_parameters()

gc.collect()
torch.cuda.empty_cache()
benchmark_metrics = launch_benchmark(train_loader, val_loader, test_loader, benchmark_run, logger = logger)
gc.collect()
torch.cuda.empty_cache()

save_benchmark_run(benchmark_run, benchmark_metrics)

