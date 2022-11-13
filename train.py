import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import torchvision
from torch.utils.data import DataLoader, Dataset
from skimage import io, color
import pandas as pd
import os
from tqdm import tqdm

# Custom data loader class to return image and label 
class SceneData(Dataset):
    """
    A class used to represent the Scene Data 

    ...

    Attributes
    ----------
    annotations_csv: str
        a formatted string to print out what the animal says
    root_dir : str
        the name of the animal
    transform : str
        the sound that the animal makes
    index : int
        index of the sample 
    
    Methods
    -------
    __len__(self)
        return number of samples in annotations_csv
    
    __getitem__(self, index)
        return image tensor and label 
    """

    def __init__(self, annotations_csv, root_dir, transform=None):
        self.annotations = pd.read_csv(annotations_csv)
        self.root_dir = root_dir
        self.transform = transform
                    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        label = torch.tensor(int(self.annotations.iloc[index, 1]))
        if self.transform:
            image = self.transform(image)
        return [image, label]

#function to check accuracy of data loaders test and train
def accuracy(loader, model, device):
    """
    Prints what the accuracy and number of correct predicitons.

        Parameters
        ----------
        loader : SceneData type
            data loader 
        
        model : 
            model to test the loader data on to calculate metric accuracy

        Returns
        ----------

        returns accuracy
        """
    correct = 0
    samples = 0

    # set model to evalutation mode for inference 
    model.eval()
    
    # set torch.no_grad to perform inference without gradient calculation
    with torch.no_grad():
        for data, targets in tqdm(loader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            _, predictions = scores.max(1)
            correct += (predictions == targets).sum()
            samples += predictions.size(0)
        #print("Correct Predictions: {}, Total Samples: {}, Accuracy: {}".format(correct, samples, int(correct) / int(samples)))
    model.train()
    return int(correct) / int(samples)



def train(epochs, bs, device):
    """
    Trains the model and saves the best model. Best model is defined as the one with highest test accuracy

        Parameters
        ----------
        epochs : int
            Number of epochs to train the model, default 10
        bs : int
            Batch Size for training, default 5
        device : "cuda" or "cpu"
            Device to train the model on, default "cpu"

        """

    # Load pre-trained alexnet model for fine-tuning
    alexnet = models.alexnet(pretrained=True) 

    # change number of output nodes for the classifier (fc layers) to number of classes
    # number of classes for this classifcication problem is 6
    alexnet.classifier[6] = nn.Linear(4096, 6)
    alexnet.to(device)
    
    learning_rate = 0.001
   
    # Transform the images by resizing it to 224x224 images 
    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # use custom data loader to load data 
    data = SceneData(annotations_csv="data/labels.csv",
                          root_dir="data/images",
                          transform = transform_img)

    #split data to train and test based on 80-20 split
    train_data, test_data = torch.utils.data.random_split(data, [112, 28])
    train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=bs, shuffle=True)

    #intialise loss criterion and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(alexnet.parameters(), lr=learning_rate) 
    
    data, targets = next(iter(train_loader))
    max_accuracy = 0

    for epoch in tqdm(range(epochs)):
        losses = []
        with tqdm(total=len(train_loader)) as pbar:
            for idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)
                alexnet.to(device)

                #forward prop
                scores = alexnet(data.to(device))
                loss = criterion(scores, targets)
                losses.append(loss)

                #back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
        print("Loss at epoch {} is {}".format(epoch, sum(losses) / len(losses)))

        # calculate accuracy for train and test data for every epoch
        train_accuracy = accuracy(train_loader, alexnet,device)
        test_accuracy = accuracy(test_loader, alexnet,device)
        print(test_accuracy)
        if test_accuracy>max_accuracy:
            max_accuracy = test_accuracy 
            best_model = alexnet
        
    torch.save(best_model, "models/model.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for Model')
    

    parser.add_argument("--epochs", help="number of Epochs the model should train on", default=10,type=int)
    parser.add_argument("--cuda", help="if used GPU will be used for training", action='store_true')
    parser.add_argument("--batch_size", help="batch size for training", default=5, type=int)
    args = parser.parse_args()
    epochs = args.epochs
    if args.cuda:
      print("Using GPU")
      device = "cuda"
    else :
      print("Using CPU")
      device = "cpu"
    bs = args.batch_size

    train(epochs, bs, device)

