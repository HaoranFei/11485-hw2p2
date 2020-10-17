import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from model import MyNetwork
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
from PIL import Image 
from sklearn import metrics

train_folder = "../data/classification_data/train_data"
val_folder = "../data/classification_data/val_data"
test_folder = "../data/classification_data/test_data"
veri_val = "../data/verification_pairs_val.txt"
veri_test = "../data/verification_pairs_test.txt"
veri_dir = "../data/"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.enabled = False

class VerificationDataset(Dataset):
    def __init__(self, is_test):
        self.source = None
        self.is_test = is_test
        if is_test:
            self.source = pd.read_csv(veri_test, delimiter=" ", header=None, names=["first_image", "second_image"])
        else:
            self.source = pd.read_csv(veri_val, delimiter=" ", header=None, names=["first_image", "second_image", "is_similar"])

    def __len__(self):
        return len(self.source.index)

    def __getitem__(self, idx):
        if self.is_test:
            return transforms.ToTensor()(Image.open(veri_dir + self.source.loc[idx, "first_image"])), \
                   transforms.ToTensor()(Image.open(veri_dir + self.source.loc[idx, "second_image"]))
        else:
            return transforms.ToTensor()(Image.open(veri_dir + self.source.loc[idx, "first_image"])), \
                   transforms.ToTensor()(Image.open(veri_dir + self.source.loc[idx, "second_image"])), \
                   self.source.loc[idx, "is_similar"]

def load_training_data(batch_size=200):
    folder = datasets.ImageFolder(train_folder, transform=transforms.ToTensor())
    num_classes = len(folder.classes)
    train_dataloader = DataLoader(folder, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    return train_dataloader, num_classes

def load_validation_data(batch_size=200):
    folder = datasets.ImageFolder(val_folder, transform=transforms.ToTensor())
    val_dataloader = DataLoader(folder, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    return val_dataloader

def load_test_data(batch_size=200):
    folder = datasets.ImageFolder(test_folder, transform=transforms.ToTensor())
    test_dataloader = DataLoader(folder, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    return test_dataloader

def load_verification_validation_data(batch_size=200):
    ds = VerificationDataset(is_test=False)
    veri_val_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    return veri_val_dataloader

def load_verification_test_data(batch_size=200):
    ds = VerificationDataset(is_test=True)
    veri_test_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    return veri_test_dataloader

def verify_val_set(net, veri_val_dataloader, epoch=None):
    scores_list = []
    labels_list = []
    for batch_img_1, batch_img_2, batch_labels in veri_val_dataloader:
        batch_img_1, batch_img_2, batch_labels = batch_img_1.to(device), batch_img_2.to(device), batch_labels.to(device)
        feature_reps_1 = net.forward(batch_img_1)
        feature_reps_2 = net.forward(batch_img_2)
        cos = nn.CosineSimilarity(dim=1)
        output = cos(feature_reps_1, feature_reps_2)
        output = output.cpu().detach().numpy()
        batch_labels = batch_labels.cpu().detach().numpy()
        scores_list.append(output)
        labels_list.append(batch_labels)
    auc = metrics.roc_auc_score(np.concatenate(labels_list), np.concatenate(scores_list))
    if epoch != None:
        print("Verifying at epoch {}. The AUC score is {}".format(epoch, auc))
    else:
        print("Verifying the model. The AUC score is {}".format(auc))

def verify_test_set(net, veri_test_dataloader):
    scores_list = []
    for batch_img_1, batch_img_2 in veri_test_dataloader:
        batch_img_1, batch_img_2 = batch_img_1.to(device), batch_img_2.to(device)
        feature_reps_1 = net.forward(batch_img_1)
        feature_reps_2 = net.forward(batch_img_2)
        cos = nn.CosineSimilarity(dim=1)
        output = cos(feature_reps_1, feature_reps_2)
        output = output.cpu().detach().numpy()
        scores_list.append(output)
    scores = np.concatenate(scores_list)
    in_df = pd.read_csv(veri_test, delimiter=" ", header=None, names=["first_image", "second_image"])
    out_df = in_df.copy(deep=True)
    out_df.columns = ["Id", "Category"]
    for i in out_df.index:
        out_df.loc[i, "Id"] = in_df["first_image"][i] + " " + in_df["second_image"][i]
        out_df.loc[i, "Category"] = scores[i]
    out_df.to_csv("./submit.csv", index=False)

def validate(net, val_dataloader, epoch):
    net.eval()
    num_correct = 0
    num_total = 0
    for minibatch, batch_labels in val_dataloader:
        minibatch, batch_labels = minibatch.to(device),batch_labels.to(device)
        y_hat = net.forward(minibatch)
        pred = torch.argmax(y_hat, dim=1)
        num_total += minibatch.shape[0]
        num_correct += torch.sum((pred == batch_labels).int()).item()
    print("Validating at epoch {}. Accuracy is {}".format(epoch, num_correct / num_total))

def test(net, test_dataloader):
    net.eval()
    num_correct = 0
    num_total = 0
    for minibatch, batch_labels in test_dataloader:
        minibatch, batch_labels = minibatch.to(device),batch_labels.to(device)
        y_hat = net.forward(minibatch)
        pred = torch.argmax(y_hat, dim=1)
        num_total += minibatch.shape[0]
        num_correct += torch.sum((pred == batch_labels).int()).item()
    print("Testing network. Accuracy is {}".format(num_correct / num_total))


# length of lr_schedule, if not none must equal to epoch
def train(net, train_dataloader, val_dataloader, batchsize=200, epoch=5, lr=0.15, lr_decay=0.85):
    loss = nn.CrossEntropyLoss()
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-5)
    lmbda = lambda epoch: lr_decay
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    start_time = time.time()

    for i in range(epoch):
        cur_time = time.time()
        elapse = cur_time - start_time
        print("At beginning of epoch {}. Total time elapsed is {} seconds".format(i, elapse))
        num_correct = 0
        num_total = 0
        total_loss = 0
        counter = 0
        for minibatch, batch_labels in train_dataloader:
            minibatch, batch_labels = minibatch.to(device),batch_labels.to(device)
            y_hat = net.forward(minibatch)
            pred = torch.argmax(y_hat, dim=1)
            num_total += minibatch.shape[0]
            num_correct += torch.sum((pred == batch_labels).int()).item()
            XELoss = loss(y_hat, batch_labels)
            total_loss += XELoss.item()
            XELoss.backward()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
        validate(net, val_dataloader, i)
        net.train()    
        scheduler.step()
        print("At Epoch {}. Average Loss is {}. Accuracy is {}".format(i, total_loss / counter, num_correct / num_total))
        torch.save(net.state_dict(), "./weights/epoch_{}.pt".format(i))

if __name__ == "__main__":
    train_dataloader, num_classes = load_training_data()
    val_dataloader = load_validation_data()
    test_dataloader = load_test_data()
    net = MyNetwork(num_classes).network
    net.load_state_dict(torch.load("./weights/archives/40_epoch.pt"))
    net.to(device)
    print(net)
    #train(net, train_dataloader, val_dataloader, epoch=20, lr_decay=0.9)
    #test(net, test_dataloader)

    veri_val_dataloader = load_verification_validation_data();
    veri_test_dataloader = load_verification_test_data();
    net_feature_extractor = nn.Sequential(*list(net.children())[:-1])
    verify_val_set(net_feature_extractor, veri_val_dataloader)
    verify_test_set(net_feature_extractor, veri_test_dataloader)

    



    














