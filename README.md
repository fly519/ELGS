# Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations
Code for the paper: Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations


## Introduction
we propose one novel model for point cloud semantic segmentation, which exploit the local and global structures within the point cloud based on the contextual point representations. Specifically, we enrich each point representation by performing one novel gated fusion on the point itself and its contextual points. Afterwards, based on the enriched representation, we propose one novel graph pointnet module (GPM), relying on the graph attention block (GAB) to dynamically compose and update each point representation within the local point cloud structure. Finally, we resort to the spatial-wise and channel-wise attention strategies to exploit the point cloud global structure and thereby yield the semantic label for each point.


## Data download and process

We provide the processed files, you can download S3DIS data <a href="https://1drv.ms/u/s!AjxFyWxg5usOajIvRkNnDLOnT3M?e=mmhCMf">here</a>  . To prepare your own S3DIS Dataset HDF5 files, refer to <a href="https://github.com/charlesq34/pointnet">PointNet</a>, you need to firstly  download <a href="http://buildingparser.stanford.edu/dataset.html">3D indoor parsing dataset version 1.2</a> (S3DIS Dataset) and convert original data to data label files by 

```bash
python collect_indoor3d_data.py
```

Finally run

```bash
python gen_indoor3d_h5.py
```

to downsampling and generate HDF5 files. You can change the number of points in the downsampling by modify this file.

## Model Training and Testing

The code is tested under TensorFlow 1.9.0 GPU version, Python 2.7.5, CUDA 9.0 and cuDNN 7.6.0 on Ubuntu 16.04. Here are some dependencies.

- `tensorflow-gpu` (1.9.0)
- `python` (2.7)
- `h5py`
- `numpy`
- `sklearn`

#### Compile TF operators

1. Find your tensorflow include path and cuda installation path.

```bash
python
import tensorflow as tf
tf.__path__
```

2. modify complie files: `tf_grouping_compile.sh`,`tf_sampling_compile.sh`and `tf_interpolate_compile.sh`.
3. Compile the shared libraries. 
```bash
cd tf_ops/3d_interpolation
./ tf_interpolate_compile.sh

cd tf_ops/grouping
./ tf_grouping_compile.sh

cd tf_ops/sampling
./ tf_sampling_compile.sh
```

Refer to <a href="https://github.com/charlesq34/pointnet2">PointNet++</a> for more details.


#### Training

When you have finished download processed data files or have prepared HDF5 files by yourself, to fill in your data path in the `train.py`. Then start training by:

```bash
cd models
python train.py
```


For S3DIS dataset, we tested on the area 5 by default. To get 6-fold results, run:

```bash
for((i=1;i<=6;i++)) \
do \
  python train_areas.py   --test_area  ${i}  --log_dir log/Area${i}  > train_and_test_area${i}.out 2>&1	 \
done \
```

you will get six models, each one of them is trained on five areas and tested on the other area.


#### Testing

After training, you can test model by:

```bash
python test.py --ckpt  your_ckpt_file  --ckpt_meta your_meta_file
```

Note that the `best_seg_model` chosen by `test.py` is only depend on overall accuracy(OA), maybe mIoU value is not the highest. Because   the overall accuracy is not necessarily proportional to the mean IoU. You can test all saved model by:

```bash
python test_all_models.py
```

The operation of segment rooms in test set is already included in this file. We use the area5 by default as the test set. You can modify it in your own code.

We provide our trained model. When you finish your Data Processing, you can test our model by: 

```bash
python test.py --ckpt trained_model/best_seg_model.ckpt  --ckpt_meta trained_model/best_seg_model.ckpt.meta
```


## Citation
```
@inproceedings{fly519,
  title={Exploiting Local and Global Structure for Point Cloud Semantic Segmentation with Contextual Point Representations},
  author={Xu Wang, Jingming He and Lin Ma},
  booktitle={NeurIPS},
  year={2019},
}
```
