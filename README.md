# human-activity-recognition 
CS 230 project on Human activity recognition for healthcare applications

This Repo has 3 models -

1) C3D - Shallow 3D CNN architecture
2) ResNet 3D - Deep 3D CNN architecture
3) Efficient 3D CNN - Resource efficient architectures

## Steps to reproduce results in report

### Install the requirements -
* [PyTorch 1.0.1.post2](http://pytorch.org/)
* OpenCV
* FFmpeg, FFprobe
* Python 3
* Resnet3D/Installation.ipynb
### Preprocess data
Preprocess NTU RGB+D data
```bash
Resnet3D/Data_preprocessing.ipynb
```
Preprocess our own self-collected data
```bash
Efficient-3DCNNs/Our_test_set_preprocessing.ipynb
```
### Generate jpg, json files and n_frames
Use the following file for generating jpg and json files
```bash
Resnet3D/genjpgandjson.sh
```
To generate n_frames -
```bash
python Efficient-3DCNNs/utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```
### Visualization of data pre-processing pipeline
For both NTU RGB+D dataset and our own data
```bash
Efficient-3DCNNs/Data_visualization.ipynb
```
### To launch training of all models
```bash
python Efficient-3DCNNs/sweep_training.py
```
### To launch validation of all models using NTU RGB+D dataset
```bash
python Efficient-3DCNNs/sweep_validation.py
```
### To launch test set with our self-collected data
```bash
python Efficient-3DCNNs/sweep_test.py
```
### To evaluate metrics such as accuracy, precision, recall, confusion matrix
```bash
Efficient-3DCNNs/Evaluate_metrics.ipynb/
```
### To sweep hyperparameters such as learning rate, no of epochs
```bash
Efficient-3DCNNs/Hyperparameters.ipynb/
```
## Model structure and no of FLOPS, parameters of each model
Refer to saved images to see the model structure - Efficient-3DCNNs/c3d, resnet18, resnet50, resnet101, resnext101, mobilenetv2, shufflenetv2, squeezenet

## Paper Results
### Accuracy/loss/learning rate of training and validation set for a sample model (ResNeXt-101)
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/loss.png" align="left" width="400" title="Loss of training and validation for a sample model (ResNeXt-101)" /><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/accuracy.png" align="right" width="400" title="Accuracy of training and validation for a sample model (ResNeXt-101)" /></p>

<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/learning_rate.png" align="left" width="400" title="Results of pre-training for action recognition task" /></p>

### Results of pre-training for action classification task
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/pretrained.png" align="middle" width="900" title="Results of pre-training for action recognition task" /></p>

### Effect of model depth on accuracy
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/model_depth.png" align="middle" width="500" title="Effect of model depth on accuracy for ResNet 3D" /></p>

### Confusion matrix for ResNet-101
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/confusion_matrix.png" align="middle" width="900" title="Confusion Matrix for ResNet-101 on 9 action classes" /></p>

### Saliency map for a few action classes
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/saliency_map.png" align="middle" width="900" title="Saliency Map for ResNet-101" /></p>

### Results on our own dataset
<p align="center"><img src="https://github.com/reetikaag/human-activity-recognition/blob/main/Efficient-3DCNNs/result_images/our_dataset_results.png" align="left" width="900" title="Results of our own dataset" /></p>
