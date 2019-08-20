# ADLD
This repository implements the training and testing of ADLD for [Weakly-Supervised Unconstrained Action Unit Detection via Latent Feature Domain](https://arxiv.org/pdf/1903.10143.pdf). The repository offers the implementation of the paper in PyTorch

# Getting Started
## Installation
- This code was tested with PyTorch 0.4.0 and Python 2.7
- Clone this repo:
```
git clone https://github.com/ZhiwenShao/ADLD
cd ADLD
```

## Datasets
Put [BP4D](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html) and [EmotioNet](http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html) into the folder "dataset" following the paths shown in the list files of the folder "data/list"

## Preprocessing
- Conduct similarity transformation for face images:
  - We provide the landmarks annotated using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for EmotioNet [here](https://sjtueducn-my.sharepoint.com/:f:/g/personal/shaozhiwen_sjtu_edu_cn/EtsjeJcurFpMpgftne6a8bMBTQcky9klDP-Js_0k2M7T3g?e=2ZrFuw). Each line in the landmark annotation file corresponds to 49 facial landmark locations (x1,y1,x2,y2...). Put these annotation files into the folder "dataset"
  - An example of processed image can be found in the folder "data/imgs/EmotioNet/optimization_set/N_0000000001/" 
```
python dataset/face_transform.py
```
- Compute the weight of the loss of each AU in the BP4D training set:
  - The AU annoatation files should be in the folder "data/list"
```
python dataset/write_AU_weight.py
```

## Training
- Train a model without using target-domain pseudo AU labels:
```
python train.py --mode='weak'
```
- Train a model using target-domain pseudo AU labels:
```
python train.py --mode='full'
```

## Testing
- Test a model trained without using target-domain pseudo AU labels:
```
python test.py --mode='weak'
```
- Test a model trained using target-domain pseudo AU labels:
```
python test.py --mode='full'
```

## Citation
If you use this code for your research, please cite our paper:
```
@article{shao2019weakly,
  title={Weakly-Supervised Unconstrained Action Unit Detection via Latent Feature Domain},
  author={Shao, Zhiwen and Cai, Jianfei and Cham, Tat-Jen and Lu, Xuequan and Ma, Lizhuang},
  journal={arXiv preprint arXiv:1903.10143},
  year={2019}
}

```
