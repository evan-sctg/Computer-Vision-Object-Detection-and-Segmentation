# Computer-Vision-Object-Detection-and-Segmentation
Computer Vision Object Detection and Segmentation using Pytorch, TorchVision, TensorBoard


# Penn-Fudan Pedestrian Detection and Segmentation

This project demonstrates object detection and instance segmentation on the Penn-Fudan Pedestrian Database using PyTorch and TorchVision. It fine-tunes a pre-trained Mask R-CNN model on the Penn-Fudan dataset, which contains 170 images with 345 instances of pedestrians.


![alt text](https://github.com/evan-sctg/Computer-Vision-Object-Detection-and-Segmentation/blob/master/pedestrians.png?raw=true)

## Overview

The training script, `train.py`, performs the following tasks:

1. **Data Preparation**: Loads and prepares the Penn-Fudan Pedestrian Database for training and evaluation.
2. **Model Customization**: Customizes the head of a pre-trained Mask R-CNN model for the specific number of classes in the dataset.
3. **Training**: Trains the customized Mask R-CNN model using the prepared dataset.

The evaluation script, `eval.py`, performs the following tasks:
1.  **Evaluation**: Evaluates the trained model on a test set and visualizes the results, including bounding box detection and instance segmentation masks.

## Requirements

- Python 3.6+
- PyTorch
- TorchVision
- PyCocoTools
- Matplotlib
- TensorBoard (for visualization)

## Setup

1. Clone the repository

2.  Install the required packages


## Usage

1. Run the training script:
python train.py



2. During training, TensorBoard logs will be saved to the `runs/PennFudanPed` directory. You can visualize the training progress using TensorBoard:
tensorboard --logdir=runs



3. After training, the model weights will be saved as `model.pth`.

4. Run the evaluation script on a test image and display the output with bounding boxes and segmentation masks. You can visualize the output using TensorBoard:
python eval.py

## Customization

You can customize the project to suit your needs by modifying the `train.py` or `eval.py` script. For example, you can:

- Change the pre-trained model used for fine-tuning
- Modify the hyperparameters (e.g., learning rate, batch size)
- Experiment with different data augmentation techniques
- Modify the Confidence Threshold
- Integrate additional evaluation metrics

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments


This project was inspired by the seminal work Mask R-CNN [https://arxiv.org/pdf/1703.06870.pdf)](https://arxiv.org/pdf/1703.06870.pdf) and utilizes code from the PyTorch and TorchVision repositories.
This README file provides an overview of the project, including its purpose, requirements, setup instructions, usage guidelines, customization options, license information, and acknowledgments. It aims to give potential employers or collaborators a clear understanding of the project's functionality and how to run and modify it. Feel free to customize the content further based on your specific needs and preferences.
Special thanks to the PyTorch team for providing an excellent deep learning framework and the open-source community for their valuable resources and contributions.
