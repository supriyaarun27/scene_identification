
import argparse
import torch
import torchvision.transforms as transforms
from skimage import io

def inference(img_path,model_path,device):
    """
    Runs inference on an image and prints the class.

        Parameters
        ----------
        img_path : string
            path to the image to run inference on 
        
        model_path : 
            model to run inference
        
        device :
            runs inference on gpu by default

        Returns
        ----------

        returns accuracy
        """
    model = torch.load(model_path)

    transform_img = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    labels = {0: "Buildings", 1: "Forests", 2: "Mountains", 3: "Glacier", 4: "Sea", 5: "Street"}

    input_image = io.imread(img_path)
    transf_img = transform_img(input_image)
    output = model(transf_img.unsqueeze(0).to(device=device))
    _, pred = output.max(1)
   
    print("Model Prediction: {}".format(labels[pred.item()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Script for Model')
    
    parser.add_argument("--image_path", help="path to image for inference")
    parser.add_argument("--model_path", help="path to .pth model for inference")
    args = parser.parse_args()
    image_path = args.image_path
    device = "cuda"
    model_path = args.model_path
    inference(image_path, model_path, device)
  