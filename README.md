# COS-Former: Hierarchical Fusion Transformer for Camouflaged Object Segmentation
## Prerequisites
The whole training process can be done on  RTX3090 + Pytorch 1.12

## Datasets
### Training Set
We use the training set of [COD10K]（https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view） and [CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)to train our model. 

### Testing Set
We test our model on the testing set of [COD10K]（https://drive.google.com/file/d/1vRYAie0JcNStcSwagmCq55eirGyMYGm5/view） and  [CAMO](https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)

## Training
Download the pretrained transformer [backbone](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) on ImageNet. 


## Testing
```
 test.py
```
## Contact
If you have any questions, feel free to email: 850992462@qq.com

