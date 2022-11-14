# Scene Identification

## Introduction

The code in this repository can do the following:
1. Fine-tune pre-trained AlexNet model for scene identification
2. Run inference for any image on the trained model 

The images are annotated and belong to 6 possible classes:
    1. Buildings
    2. Forests
    3. Mountains
    4. Glacier
    5. Sea
    6. Street

### Folder Structure 

```
scene_identification
│   README.md
│   inference.py
|   train.py
|   requirements.txt    
│
└───data
│   │   labels.csv
│   │
│   └───images
│       │   0.jpg
│       │   1.jpg
│       │   ...
│   
└───inference_images
|   │   test1.jpg
|   │   test2.jpg
|   |   ...
└───models
|   | model.pth


```

- data folder :
    - labels.csv contains annotations for all the images in the "images" folder
    - images folder contains 140 images for training and testing 
- inference folder 
    - containes 5 test images (not belonging to the images folder) which you can use for checking the inference on the model 
- models folder 
    - contains the trained .pth model 
    - this is also the default location where the models are stored after training 

- inference.py
    - script to run inference on the model 
- train.py
    - script to train the model 

## Approach 

To solve this problem of scene identification, I have picked a transfer learning approach. 

#### What is Transfer Learning?

Using an existing model trained on a different domain/task and use it for a new domain/task. 
##### Advantages

- Save Time 
- Improved performance 
- Better generalisability 

#### How is it being used here?

In this project I have used the ALexNet pretrained network as a feature extracter and modified the classifier network to output the probability of 6 classes. 

I have fine-tuned the pre-trained AlexNet on the 112 (80-20 split on the 140 images) images from the data folder. The model is trained end-to-end which now works as a domain adaptation network trained to now identify the type of scene in the image. 

#### AlexNet

AlexNet is pre-trained on images from the ImageNet LSVRC-2010 dataset, which has over 1.2 million images divided to 1000 classes. The Network has 5 convolutional layers each followed by a ReLu, two fully connected layers and a softmax classifier. AlexNet has 60 million paramaters.

This is what the arhitechture looks like : 

![Alt text](ref_img/arch.png?raw=true "AlexNet Architecture")

I have modifed the final classification layer to have 6 output nodes for this classification task.

#### Dataset

For simplifiying training time, I have only considered 140 images for training and testing. The script takes an 80-20 train-test split by default. 

These are some of the images in the dataset:

![Alt text](ref_img/0.jpg? "AlexNet Architecture")
![Alt text](ref_img/1.jpg? "AlexNet Architecture")
![Alt text](ref_img/2.jpg? "AlexNet Architecture")

The images are annotated and belong to 6 possible classes:
    1. Buildings
    2. Forests
    3. Mountains
    4. Glacier
    5. Sea
    6. Street

#### Training Paramaters

##### Loss Function

Cross Entropy Loss : Since this is a multi-class lassification task ( with 6 classes ) I have picked the cross entorpy loss as the cost function. 

##### Optimiser

Adam : the adam optimiser is widely used in most deep learning approaches as it is computationally inexpensive and is supposed to be great when working with large models (such as AlexNet)

##### Evaluation Criteria 

Accuracy : Since it is a multi-class classification problem i have picked Accuracy as the evalutation metric 

## Run the code yourself

##### Create a virtual Environemnet (recommended) 

##### Install dependencies 

##### Unzip the images folder 

##### Train 

```
python train.py --epochs 10  --batch_size 5 --cuda
```
Parameters :

1. epochs : number of epochs to train the model on, default 10 
2. batch_size : batch size for training, default 5
3. cuda : if this passed, the model will train on gpu, else cpu. Default cpu. 

##### Inference

```
python inference.py --image_path "inference_images/test1.jpg
```

### Future Scope / Improvements