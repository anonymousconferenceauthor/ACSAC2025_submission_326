"""
Pre-train models to later unlearn
'This is not unlearning code'
"""
from model_builder import resnet10, resnet50
from data import *
from utils import *
from train import *
from random_sample_indices import * 
import torch
import torch.nn as nn
from torch.utils.data import Subset
import numpy as np
import argparse
import time 
from tqdm import tqdm

set_seed(42)

def main(dataset, backbone, task, num_epochs): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    unlearn_k = 0     
    unleran_ks = [16, 32, 64, 128]
    current_time = time.localtime()
    formatted_time = time.strftime("%m%d%H%M", current_time)

    model_path = "models/{}/pre_trained_models/{}_{}_{}.pt".format(dataset,dataset,task,backbone) 

    #select dataset
    num_classes=200
    trainset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='train', transform=transform_tinyimagenet)
    testset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='val', transform=transform_tinyimagenet, class_to_idx=trainset.class_to_idx)

    #generate ratain/unlearning datasets
    train_labels = trainset.targets 
    train_labels = torch.from_numpy(np.array(train_labels))
    indices_k_unlearn = torch.randperm(train_labels.shape[0])[:unlearn_k]
    print ('indices_k_unlearn : ', indices_k_unlearn)
    copy_train_labels = train_labels.clone()
    copy_train_labels[indices_k_unlearn] = -10
    indices_retain_data = (copy_train_labels != -10).nonzero(as_tuple=False)
    unlearn_dataset = Subset(trainset, indices_k_unlearn.view(-1,))
    #unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset,batch_size=batch_size,shuffle=True)
    retain_dataset = Subset(trainset, indices_retain_data.view(-1,))
    retain_loader = torch.utils.data.DataLoader(retain_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    #select backbone
    if backbone == "resnet10":
        model = resnet10(num_classes=num_classes)
    elif backbone=="resnet50":
        model = resnet50(num_classes=num_classes)
    else: 
        raise ValueError("Invalid backbone")
    

    #train model 
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                milestones=[30, 60, 90],
                                                gamma=0.2)
    criterion = nn.CrossEntropyLoss()

    pre_train(model, retain_loader, test_loader, optimizer, criterion, num_epochs, device, model_path, scheduler)
    test_acc = evaluate(model, test_loader, device)
    print("test acc:{}".format(test_acc))

    unlearn_accs = {}
    retain_accs = {}
    unlearn_mias = {} 
    train_labels = trainset.targets 
    train_labels = torch.from_numpy(np.array(train_labels))

    for unlearn_k in tqdm(unleran_ks):
        key = "{}_{}".format(dataset,str(unlearn_k)) 
        indices_k_unlearn = torch.tensor(random_indices[key]) 
        copy_train_labels = train_labels.clone()
        copy_train_labels[indices_k_unlearn] = -10
        indices_retain_data = (copy_train_labels != -10).nonzero(as_tuple=False)
        unlearn_dataset = Subset(trainset, indices_k_unlearn.view(-1,))
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset,batch_size=unlearn_k, shuffle=True)
        retain_dataset = Subset(trainset, indices_retain_data.view(-1,))
        retain_loader = torch.utils.data.DataLoader(retain_dataset,batch_size=batch_size, shuffle=True)
        _,_,unlearn_acc = eval(model, unlearn_loader, device, model_path)
        _,_,retain_acc = eval(model, retain_loader, device, model_path)
        unlearn_accs[str('k'+str(unlearn_k))] = unlearn_acc
        retain_accs[str('k'+str(unlearn_k))] = retain_acc
        mia = get_membership_attack_prob(retain_loader, unlearn_loader, test_loader, model)
        unlearn_mias[str('k'+str(unlearn_k))] = mia

    #generate report
    file_path = "logs/{}/pre_trained_models/{}_{}_{}_{}.txt".format(dataset,dataset, task, backbone,formatted_time)
    with open(file_path, 'w') as file: 
        file.write("CONFIGURATIONS\n")
        file.write('Dataset:{}\n'.format(dataset))
        file.write('Backbone_model:{}\n'.format(model.__class__.__name__))
        file.write('Loss:{}\n'.format(criterion.__class__.__name__))
        file.write('Optimizer:{}\n'.format(optimizer.__class__.__name__))
        file.write('Learning_rate:{}\n'.format(optimizer.param_groups[0]['lr']))
        file.write('Batch_size:{}\n'.format(batch_size))
        file.write('Epoch:{}\n'.format(num_epochs))       
        file.write('\n')
        file.write("PERFOMANCE\n")
        file.write("retain acc:{}\n".format(retain_accs))
        file.write('test accuracy:{}\n'.format(test_acc))
        file.write('unlearn data accuracy:{}\n'.format(str(unlearn_accs)))
        file.write('MIA:{}\n'.format(str(unlearn_mias)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train models")
    parser.add_argument('--dataset', type=str, default='TinyImageNet')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--task', type=str, default='random')
    parser.add_argument('--num_epochs', type=int, default=100)  
    args = parser.parse_args()

    dataset = args.dataset
    backbone = args.backbone
    task = args.task
    num_epochs = args.num_epochs
    
    main(dataset, backbone, task, num_epochs)

