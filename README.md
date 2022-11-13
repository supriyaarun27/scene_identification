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
|   |

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

### Approach 

To solve this problem of scene identification, I have picked a transfer learning approach. 

##### What is Transfer Learning?

