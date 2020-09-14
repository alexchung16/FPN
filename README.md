# FPN-Tensorflow
FPN Network

## compile for num and rpn 
```shell script
cd $PATH_ROOT/libs/box_utils/cython_utils
python setup.py build_ext --inplace
```

## Download Dataset
### Pascal Voc 2012
```shell script
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
### COCO 2017
```shell script
$ wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
````
# pretrained weights
* <http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz>

## train logs

![acc](./docs/rpn_loss.png) |
|:-------------------------:|
Displayed one-state Loss on Tensorboard | 

![acc](./docs/fastrcnn_loss.png) |
|:-------------------------:|
Displayed two-stage Loss on Tensorboard | 

![acc](./docs/total_loss.png)      |  ![](./docs/learning_rate.png)
:-------------------------:|:-------------------------:
Displayed Total Loss on Tensorboard |  Displayed Accuracy on Tensorboard

![acc](./docs/acc.png) |
|:-------------------------:|
Displayed Accuracy on Tensorboard | 


## TODO