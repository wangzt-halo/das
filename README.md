# Distribution-Aware Single-Stage Models for Multi-Person 3D Pose Estimation
Code for "Distribution-Aware Single-Stage Models for Multi-Person 3D Pose Estimation".

## Installation

Follow the instructions in [MMDetection3D](https://github.com/open-mmlab/mmdetection3d):

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).
```shell
pip install openmim
mim install mmcv-full==1.3.10
```

**Step 1.** Install [MMDetection](https://github.com/open-mmlab/mmdetection).
```shell
pip install mmdet==2.14.0
```

**Step 2.** Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
```shell
pip install mmsegmentation
```

**Step 3.** Clone the DAS repository.
```shell
git clone https://github.com/wangzt-halo/das.git
cd das
```

**Step 4.** Install build requirements and then install DAS.
```shell
pip install -v -e .  # or "python setup.py develop"
```

## Prepare Data

### Download Datasets

#### CMU Panoptic
Download CMU Panoptic from [Joo et al.](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox). \
Process raw data.
```angular2html
python mytools/panoptic2coco.py --root /path/to/panoptic
```

#### MuCo-3DHP and MuPoTS-3D
Download MuCo-3DHP and MuPoTS-3D from [Moon et al.](https://github.com/mks0601/3DMPPE_POSENET_RELEASE).

#### COCO
Download COCO from [COCO Dataset](https://cocodataset.org/).


### Directory Structure
```
${ROOT}
|-- data
|   |-- panoptic
|   |   |-- 160226_haggling1
|   |   |   |-- hdImgs
|   |   |   |   |-- 00_16
|   |   |   |   |-- 00_30
|   |   |-- 160422_haggling1
|   |   |-- 160226_mafia1
|   |   |-- 160422_mafia2
|   |   |-- 160226_ultimatum1
|   |   |-- 160422_ultimatum1
|   |   |-- 160906_pizza1
|   |   |-- annotations
|   |   |   |-- train_new.json
|   |   |   |-- haggling.json
|   |   |   |-- mafia.json
|   |   |   |-- ultimatum.json
|   |   |   |-- pizza.json
|   |-- coco
|   |   |-- train2017
|   |   |-- annotations
|   |   |   |-- person_keypoints_train2017.json
|   |-- mupots
|   |   |-- TS1
|   |   |   |-- img_000000.jpg
|   |   |   |-- ...
|   |   |-- ...
|   |   |-- TS20
|   |   |-- annotations
|   |   |   |-- MuPoTS-3D.json
```

## Pretrained Model
Download the pretrained MSPN models from [MMPose](https://mmpose.readthedocs.io/en/latest/papers/backbones.html?highlight=mspn#mspn-arxiv-2019).\
Put the models in ```${ROOT}/weights/```

## Training and Evaluation
Train on CMU Panoptic dataset:
```angular2html
bash tools/dist_train.py configs/das/exp_panoptic.py 4
```

Evaluate on CMU Panoptic dataset:
```angular2html
bash tools/dist_test.py configs/das/exp_panoptic.py work_dirs/exp_panoptic/latest.pth 4 --eval mpjpe
```