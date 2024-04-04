import torch
from torchvision.models import mobilenet_v2
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tarfile
import os
import pandas as pd
import shutil
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename='training.log', level=logging.INFO)

class CUBModel():
    def __init__(self, device = "cpu", lr = 0.001, weight_decay = 0.0001, batch_size = 32, momentum = 0.9, image_size = 224) -> None:
        self.device = device
        logging.info(f"Device used is : {str(device)}")
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.momentum = momentum
        self.image_size = image_size
        self.batch_size = batch_size

        self.model = mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.num_classes = 200

        # Get the number of output features of the last layer
        num_features = self.model.classifier[-1].out_features

        # Define your custom layers
        self.custom_layers = nn.Sequential(
            nn.Linear(num_features, 512),  # Add a linear layer
            nn.ReLU(inplace=True),         # Add ReLU activation
            nn.Linear(512, self.num_classes),   # Add another linear layer for classification
        )

        # Concatenate the MobileNetV2 model with custom layers
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children()) + list(self.custom_layers.children()))
        self.model = self.model.to(self.device)

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(int(self.image_size/0.875)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def train(self, train_data_path, test_data_path, num_epochs=40):
        train_dataset = datasets.ImageFolder(train_data_path, transform=self.train_transform)
        test_dataset = datasets.ImageFolder(test_data_path, transform=self.test_transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        logging.info("Starting training the model...")
        stop_cnt = 0
        test_losses, train_losses, train_acc, test_acc = [], [], [], []
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(images)

                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            # Print the average loss for the epoch
            train_losses.append(running_loss/len(train_loader))
            train_acc.append(100*correct/total)
            accuracy, test_loss = self._accuracy(test_loader)
            test_acc.append(accuracy)
            test_losses.append(test_loss)
            if len(test_losses) > 0 and test_loss > test_losses[-1]:
                stop_cnt += 1
                if stop_cnt == 3:
                    logging.info("Early stopping...")
                    test_losses.append(test_loss)
                    logging.info("Training finished")
                    self.save_model("./final_model_checkpoint.pth")
                    logging.info("Model saved")
                    self.plot_loss(train_losses, test_losses)
                    self.plot_accuracy(train_acc, test_acc)
                    return
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader):.4f}, Training Accuracy: {100*correct/total}%, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
        logging.info("Training finished")
        self.save_model("./final_model_checkpoint.pth")
        logging.info("Model saved")
        self.plot_loss(train_losses, test_losses)
        self.plot_accuracy(train_acc, test_acc)

    def plot_accuracy(self, train_acc, test_acc):
        plt.figure()
        plt.plot(train_acc, label='Training accuracy')
        plt.plot(test_acc, label='Validation accuracy')
        plt.legend(frameon=False)
        plt.savefig("accuracy.png")

    def plot_loss(self, train_losses, test_losses):
        plt.figure()
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig("loss.png")

    def _accuracy(self, test_loader):
        # Evaluation
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            running_loss = 0.0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Print the accuracy on the test set
        accuracy = 100 * correct / total
        logging.info(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy, running_loss/len(test_loader)

    def predict_from_image_path(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Forward pass through the model
        with torch.no_grad():
            output = self.forward(image)

        return output

    def predict_from_image_array(self, image):
        # Load and preprocess the image
        image = self.transform(image).unsqueeze(0)  # Add batch dimension

        # Forward pass through the model
        with torch.no_grad():
            output = self.forward(image)

        return output

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(filepath):
        model = CUBModel()
        model.model.load_state_dict(torch.load(filepath))
        return model

def extract_tarfile(tarfile_address):
    if (tarfile_address[-7:] != ".tar.gz") and (tarfile_address[-4:] != ".tgz"):
        logging.error("The file is not a tar.gz file")
        raise ValueError("The file is not a tar.gz file")
    tar = tarfile.open(tarfile_address)
    if tarfile_address[0:2] == "./":
        tarfile_address = tarfile_address[2:]
    folder_address = tarfile_address.split(".")[0]
    logging.info(f"Folder address detected for tarfile : {folder_address}")
    os.system(f'mkdir -p {folder_address}')
    tar.extractall(folder_address)
    tar.close()
    if folder_address[0] != '/' and folder_address[:4] != "home":
        folder_address = './' + folder_address
    folder_address = folder_address + "/CUB_200_2011/"
    return folder_address

def split_dataset(path : str):
    if path[-1] != "/":
        path = path + "/"

    os.system("mkdir -p ./split_dataset")
    os.system("mkdir -p ./split_dataset/train")
    os.system("mkdir -p ./split_dataset/test")

    df = pd.read_csv(path + "images.txt", sep=" ", header=None)
    df.columns = ["id", "path"]

    test_train_split = pd.read_csv(path + "train_test_split.txt", sep=" ", header=None)
    test_train_split.columns = ["id", "is_train"]


    for i in range(len(df)):
        if test_train_split["is_train"][i] == 0:
            file_path = path+"images/"+df["path"].iloc[i]
            dir_name = df["path"].iloc[i].split("/")[0]
            try:
                os.mkdir(os.path.join("./split_dataset/train", dir_name))
            except:
                pass
            destination = os.path.join("./split_dataset/train", df["path"].iloc[i])
            shutil.copy(file_path, destination)
        else:
            file_path = path+"images/"+df["path"].iloc[i]
            dir_name = df["path"].iloc[i].split("/")[0]
            try:
                os.mkdir(os.path.join("./split_dataset/test", dir_name))
            except:
                pass
            destination = os.path.join("./split_dataset/test", df["path"].iloc[i])
            shutil.copy(file_path, destination)

    return "./split_dataset/train", "./split_dataset/test"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Options for model training on CUB_200_2011 datasets'
    )
    parser.add_argument('--tarfile_address', type=str, required=True, help='Address of the tar file containing the dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='base learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='batch size for training')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay for SGD')
    parser.add_argument('--image_size', type=int, default=224, help='image\'s size for transforms')
    parser.add_argument('--gpu_id', type=int, default=0, help='choose one gpu for training')
    args = parser.parse_args()

    device = torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    os.system('touch training.log')
    mymodel = CUBModel(device=device, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, momentum=args.momentum, image_size=args.image_size)
    folder_address = extract_tarfile(args.tarfile_address)
    logging.info("Dataset extracted...")
    train_path, test_path = split_dataset(folder_address)
    logging.info("Dataset splitted...")
    mymodel.train(train_path, test_path, args.epochs)
    os.system('rm -rf CUB_200_2011 2> /dev/null')
    os.system('rm -rf split_dataset 2> /dev/null')
