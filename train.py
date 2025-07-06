import torch
import os
from tqdm import  tqdm
from utils import *

def pre_train(backbone, train_loader, val_loader, optimizer, criterion, num_epochs, device, model_path, scheduler=None):
    best_train_loss = 10
    for epoch in tqdm(range(num_epochs)):
        backbone.train()
        train_total_correct = 0 
        train_total_loss = 0
        train_total_samples = 0
        
        for images, labels in tqdm(train_loader): 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = backbone(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()*labels.size(0)
            _, predicted = torch.max(pred.data,1)

            train_total_correct += (predicted==labels).sum().item()
            train_total_samples += labels.size(0)
        
        train_acc = train_total_correct/train_total_samples
        train_loss = train_total_loss/train_total_samples
        print(f"Epoch {epoch+1}/{num_epochs},train loss: {train_loss:.4f}, train acc:{train_acc:.4f}")
        
        if train_loss<best_train_loss: 
            directory = os.path.dirname(model_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(backbone.state_dict(), model_path)
            best_train_loss = train_loss
            print("best model at epoch{}".format(epoch))
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")     

def eval(model, loader, device, model_path):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    total_correct = 0 
    total_samples = 0 
    predictions = []
    answers = []

    with torch.no_grad():
        for images, labels in loader:
            imgaes, labels = images.to(device), labels.to(device)
            
            pred = model(imgaes)
            _,prediction = torch.max(pred.data,1)
            total_correct += (prediction == labels).sum().item()
            total_samples += labels.size(0)
            predictions.append(prediction)
            answers.append(labels)
    accuracy = total_correct/total_samples
    predictions = to_list(predictions)
    answers = to_list(answers)

    return predictions, answers, accuracy


