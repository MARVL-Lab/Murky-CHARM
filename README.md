# Murky-CHARM
This is the official Github repo of Murky-CHARM: Coral Health Assessment framework using Robots in Murky environments. We provide the implementation of our collected murky coral segmentation dataset and a pretrained segmentation model on our murky coral segementation data. We finetune the Coralscapes pre-trained model in this repo and provide the scripts to perform finetuning and inference on our murky coral segmentation dataset in the MarvlCoralScapes folder.

To clone this repo:
```
git clone --recurse-submodules -j8 https://github.com/MARVL-Lab/Murky-CHARM.git
```

## Murky Coral Segmentation Dataset and Finetuned Murky Coral Segmentation Model
Please refer to the MarvlCoralScapes folder for more details. We provide the scripts we used for finetuning and visualization.

Here are the links to download the dataset and weights for the model:

[Murky Coral Segementation Dataset](https://drive.google.com/file/d/1qtkDf73nYUtCg39GXdROgDOnztnZKE-j/view?usp=sharing)

[Finetuned Murky Coral Segmentation Model](https://drive.google.com/file/d/1poVef4G1vSdkLMk4tU0Y_3MPN51mpYlX/view?usp=sharing)

## Citation
If you find this project useful, please consider citing Murky-CHARM:
```bibtex
@article{tan2025murkycharm,
        title={Murky-CHARM: Coral Health Assessment using Robots in Murky environments}, 
        author={Yu Xiang Tan and Marcel Bartholomeus Prasetyo and Shrutika Thengane and Jun Jie Hoo and Malika Meghjani},
        year={2025},
  }
```