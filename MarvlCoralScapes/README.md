<div align="center">
<h1>Murky-CHARM Segmentation</h1>
</div>

We build our segmentation model upon the Coralscapes trained Segformer model. We modified some parts of the inference pipeline to convert the coralscape classes to our murky dataset classes. Thus, to run finetuning and inference on the murky dataset, please clone this repo and follow the instructions below. Since our finetuning and inference scripts were modified from the coralscapes scripts, please also follow the Coralscapes installation method below then follow the Murky-CHARM Segmentation instructions to perform training or inference with the Murky-CHARM dataset. 

## Murky-CHARM Instructions

### Finetune

To finetune the model on the murky dataset, run the following command:
```bash
python nbs/fine_tune.py
```
Before running the script, make sure to update the following paths in the code:
In config/base.yaml file change the following path 
```bash
root: "<input_dataset_path>"
```
and if you want to change the base model, change the model name in fine_tune.py file "config_path='configs/model_name.yaml'".

To modify the batch size, learning rate and run name, replace the parameters listed in line 38 of nbs/fine_tune.py

### Inference

This inference script will output the score of the given model in a metrics.txt file from the directory that the command is ran on.
If you want to do the inference with the pretrained model on the murky dataset, run the following command:
```bash
python nbs/inference.py --is_coralscape
```
Before running the script, make sure to update the following paths in the code:
In config/base.yaml file change the following path 
```bash
root: "<input_dataset_path>"
```
And if you want to do the inference on your finetuned model, run the following command:
```bash
python nbs/inference.py --model_checkpoint <model_checkpoint_path>
```
Before running the script, make sure to update the following paths in the code:
In config/base.yaml file change the following path 
```bash
root: "<input_dataset_path>"
```

In the inference process, if you want to save the predicted images, uncomment the code block in nbs/inference.py from line 141 onwards and change the output inference path

### Citation
If you find this project useful, please consider citing Murky-CHARM:
```bibtex
@article{tan2025murkycharm,
        title={Murky-CHARM: Coral Health Assessment using Robots in Murky environments}, 
        author={Yu Xiang Tan and Marcel Bartholomeus Prasetyo and Shrutika Thengane and Jun Jie Hoo and Malika Meghjani},
        year={2025},
  }
```

Please also consider citing Coralscapes:
```bibtex
@misc{sauder2025coralscapesdatasetsemanticscene,
        title={The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs}, 
        author={Jonathan Sauder and Viktor Domazetoski and Guilhem Banc-Prandi and Gabriela Perna and Anders Meibom and Devis Tuia},
        year={2025},
        eprint={2503.20000},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2503.20000}, 
  }
```

<div align="center">
<h1>The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs</h1>
</div>


### Installation

We recommend that you first set up a Python virtual environment using a package manager. In our case we utilize [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) because it’s lightweight, fast and easy to use. However, you are free to use any other managers (e.g. [conda](https://docs.conda.io/en/latest/)) by adjusting the commands accordingly.

```bash
git clone https://github.com/eceo-epfl/coralscapesScripts 
cd coralscapesScripts
micromamba env create -f environment.yml # Setup the environment by installing required packages
micromamba activate coralscapes 
```

### Dataset
There are two ways to explore the dataset within your code.

- The first approach is to clone this repository and download the dataset from [zenodo](https://zenodo.org/records/15061505). Then you can use the `Coralscapes` class to load the dataset splits given their location.
    ```python
    from coralscapesScripts.datasets.dataset import Coralscapes

    coralscapes_root_dir = "../coralscapes" # Update based on the location of your dataset 

    train_dataset = Coralscapes(root = coralscapes_root_dir, split = 'train')
    val_dataset = Coralscapes(root = coralscapes_root_dir, split = 'val') 
    test_dataset = Coralscapes(root = coralscapes_root_dir, split = 'test')

    image, label = test_dataset[42]
    ```

- Alternatively, you can use the Hugging Face version of the [dataset](https://huggingface.co/datasets/EPFL-ECEO/coralscapes). 
    ```python
    from datasets import load_dataset

    dataset = load_dataset("EPFL-ECEO/coralscapes") 

    image = dataset["test"][42]["image"]
    label = dataset["test"][42]["label"]
    ```

- Samples from the dataset with their associated colors can be visualized as follows: 
    ```python
    from coralscapesScripts.visualization import show_samples
    show_samples(train_dataset, n=2) # To visualize two images from the dataset
    ```

### Semantic Segmentation Inference
There are several options to use the models fine-tuned on Coralscapes: 
- You can use the `scripts/inference.py` script using a downloaded model checkpoint as shown below. This script utilizes a sliding window approach for the prediction, thus the prediction results may differ from the ones used in the model evaluation. In order to replicate the evaluation procedure of the specific models, use the `scripts/inference_evaluation.py` script.    
    ```bash
    cd scripts
    python inference.py \
    --inputs <path> \ # path to image or directory of images (e.g. ../img_dir/example_image.png or ../img_dir)
    --outputs <dir> \ # path to directory to save the images (e.g. ../output_dir/)
    --config <path> \ # path to a model configuration (e.g. ../configs/segformer-mit-b2.yaml)
    --model-checkpoint  <path> # path to model_checkpoint (e.g. ../model_checkpoints/segformer_mit_b2_epoch150)

- You can follow the steps at `nbs/inference.ipynb` using a downloaded model checkpoint.

- You can use the Hugging Face model checkpoints:
    - Simple Approach
        ```python 
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        from PIL import Image
        from datasets import load_dataset

        # Load an image from the coralscapes dataset or load your own image 
        dataset = load_dataset("EPFL-ECEO/coralscapes") 
        image = dataset["test"][42]["image"]

        preprocessor = SegformerImageProcessor.from_pretrained("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")
        model = SegformerForSemanticSegmentation.from_pretrained("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        inputs = preprocessor(image, return_tensors = "pt")
        outputs = model(**inputs)
        outputs = preprocessor.post_process_semantic_segmentation(outputs, target_sizes=[(image.size[1], image.size[0])])
        label_pred = outputs[0].cpu().numpy()
        ```
    
    - Sliding Window Approach 
        ```python 
        import torch 
        import torch.nn.functional as F
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        from PIL import Image
        import numpy as np
        from datasets import load_dataset
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        def resize_image(image, target_size=1024):
            """
            Used to resize the image such that the smaller side equals 1024
            """
            h_img, w_img = image.size
            if h_img < w_img:
                new_h, new_w = target_size, int(w_img * (target_size / h_img))
            else:
                new_h, new_w  = int(h_img * (target_size / w_img)), target_size
            resized_img = image.resize((new_h, new_w))
            return resized_img

        def segment_image(image, preprocessor, model, crop_size = (1024, 1024), num_classes = 40, transform=None):
            """
            Finds an optimal stride based on the image size and aspect ratio to create
            overlapping sliding windows of size 1024x1024 which are then fed into the model.  
            """ 
            h_crop, w_crop = crop_size
            
            img = torch.Tensor(np.array(resize_image(image, target_size=1024)).transpose(2, 0, 1)).unsqueeze(0)
            batch_size, _, h_img, w_img = img.size()
            
            if transform:
                img = torch.Tensor(transform(image = img.numpy())["image"]).to(device)    
                
            h_grids = int(np.round(3/2*h_img/h_crop)) if h_img > h_crop else 1
            w_grids = int(np.round(3/2*w_img/w_crop)) if w_img > w_crop else 1
            
            h_stride = int((h_img - h_crop + h_grids -1)/(h_grids -1)) if h_grids > 1 else h_crop
            w_stride = int((w_img - w_crop + w_grids -1)/(w_grids -1)) if w_grids > 1 else w_crop
            
            preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
            count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
            
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    with torch.no_grad():
                        if(preprocessor):
                            inputs = preprocessor(crop_img, return_tensors = "pt")
                            inputs["pixel_values"] = inputs["pixel_values"].to(device)
                        else:
                            inputs = crop_img.to(device)
                        outputs = model(**inputs)
    
                    resized_logits = F.interpolate(
                        outputs.logits[0].unsqueeze(dim=0), size=crop_img.shape[-2:], mode="bilinear", align_corners=False
                    )
                    preds += F.pad(resized_logits,
                                    (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2))).cpu()
                    count_mat[:, :, y1:y2, x1:x2] += 1
                
            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            preds = preds.argmax(dim=1)
            preds = F.interpolate(preds.unsqueeze(0).type(torch.uint8), size=image.size[::-1], mode='nearest')
            label_pred = preds.squeeze().cpu().numpy()
            return label_pred

        # Load an image from the coralscapes dataset or load your own image 
        dataset = load_dataset("EPFL-ECEO/coralscapes") 
        image = dataset["test"][42]["image"]

        preprocessor = SegformerImageProcessor.from_pretrained("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")
        model = SegformerForSemanticSegmentation.from_pretrained("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024")

        label_pred = segment_image(image, preprocessor, model)
        ```

- You can use the online Hugging Face gradio [demo](https://huggingface.co/spaces/EPFL-ECEO/coralscapes_demo).  


### Semantic Segmentation Fine-tuning
To fine-tune an existing model checkpoint on a custom dataset, please follow the guide at `nbs/fine_tune.ipynb`.



## Structure

### Dataset Structure
The dataset structure of the Coralscapes dataset follows the structure of the Cityscapes dataset:

```
{root}/{type}/{split}/{site}/{site}_{seq:0>6}_{frame:0>6}_{type}{ext}
```

The meaning of the individual elements is:
 - `root`  the root folder of the Coralscapes dataset. 
 - `type`  the type/modality of data, `gtFine` for fine ground truth, or `leftImg8bit` for left 8-bit images, `leftImg8bit_1080p (gtFine_1080p)` for the images (ground truth) in 1080p resolution, `leftImg8bit_videoframes` for the 19 preceding and 10 trailing video frames.
 - `split` the split, i.e. train/val/test. Note that not all kinds of data exist for all splits. Thus, do not be surprised to occasionally find empty folders.
 - `site`  ID of the site in which this part of the dataset was recorded.
 - `seq`   the sequence number using 6 digits.
 - `frame` the frame number using 6 digits. 
 - `ext`   `.png`


```



## Citation
If you find this project useful, please consider citing:
```bibtex
@misc{sauder2025coralscapesdatasetsemanticscene,
        title={The Coralscapes Dataset: Semantic Scene Understanding in Coral Reefs}, 
        author={Jonathan Sauder and Viktor Domazetoski and Guilhem Banc-Prandi and Gabriela Perna and Anders Meibom and Devis Tuia},
        year={2025},
        eprint={2503.20000},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2503.20000}, 
  }
```
