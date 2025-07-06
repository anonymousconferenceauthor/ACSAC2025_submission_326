"""
main run code for unlearning 
baselines: naive retrain, retrain unlearn data w/ incorrect labels, Amnesiac unlearning
"""
from methods.naive_retrain import *
from methods.neggrad import *
from methods.finetune import *
from methods.oracle import *
from methods.psrp import *
from methods.ssd import *
from methods.teacher import * 
from methods.unsir import *
from methods.neggrad_paramselect import * 
from methods.efmu import *
from methods.efmu_lam import *
from methods.lipschitz import *
from methods.disentangle import *
from model_builder import *
from model_builder import lenet, resnet10, resnet50, vgg11
from custom_loss import *
from data import *
from utils import *
from train import *
from random_sample_indices import * 

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import argparse
import time 
set_seed(10)

def main(experiment_name, task, method, dataset, backbone, unlearn_k, f_cls, s_cls, 
            batch_size, selection_rate, device, num_epochs, retain_percentage):
    unlearn_batch_size = batch_size
    

    if device=="0":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device=="1":
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:{}".format(device))
    current_time = time.localtime()
    formatted_time = time.strftime("%m%d%H%M", current_time)

    if task=='random':
        model_path = "models/{}/unlearned_models_{}/{}/{}_unlearnk{}_{}_{}.pt".format(dataset,task, method,  method, str(unlearn_k), backbone,formatted_time) 
        report_path = "logs/{}/unlearned_models_{}/{}/{}_unlearnk{}_{}_{}_10.txt".format(dataset,task,method, method,str(unlearn_k),backbone, formatted_time)
    elif task=='full_class':
        model_path = "models/{}/unlearned_models_{}/{}/{}_cls{}_{}_{}.pt".format(dataset,task, method,  method, str(f_cls), backbone,formatted_time) 
        report_path = "logs/{}/unlearned_models_{}/{}/{}_cls{}_{}_{}_10.txt".format(dataset,task,method, method,str(f_cls),backbone, formatted_time)
    elif task=='sub_class':
        model_path = "models/{}/unlearned_models_{}/{}/{}_cls{}_{}_{}.pt".format(dataset,task, method,  method, str(s_cls), backbone,formatted_time) 
        report_path = "logs/{}/unlearned_models_{}/{}/{}_cls{}_{}_{}_10.txt".format(dataset,task,method, method,str(s_cls),backbone, formatted_time)
    else: 
        ValueError("Invalid task")
        
    #select dataset
    if dataset == "CIFAR20":
        num_classes=20
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        trainset = CIFAR20Dataset(trainset)
        testset = CIFAR20Dataset(testset)
        original_labels = trainset.original_targets
        original_labels=torch.from_numpy(np.array(original_labels))
    elif dataset == "TinyImageNet":
        num_classes=200
        if "vit" in backbone:
            trainset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='train', transform=transform_tinyimagenet_vit)
            testset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='val', transform=transform_tinyimagenet_vit, class_to_idx=trainset.class_to_idx)
        else:
            trainset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='train', transform=transform_tinyimagenet)
            testset = TinyImageNetDataset(root_dir='./data/tiny-imagenet-200', split='val', transform=transform_tinyimagenet, class_to_idx=trainset.class_to_idx)
    else: 
        raise ValueError("Invalid dataset")

    #generate ratain/unlearning datasets/loaders
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                            num_workers=4, pin_memory=True)
    train_labels = trainset.targets 
    train_labels = torch.from_numpy(np.array(train_labels))

    if task=="random":
        key = "{}_{}".format(dataset,str(unlearn_k))
        print(key)
        indices_k_unlearn = torch.tensor(random_indices[key]) 
    elif task=="full_class":
        print('full class to unlearn:{}'.format(f_cls))
        indices_k_unlearn = (train_labels == f_cls).nonzero(as_tuple=False).view(-1)
    elif task == "sub_class":
        indices_k_unlearn = (original_labels == s_cls).nonzero(as_tuple=False).view(-1)
    else: 
        raise ValueError("Invalid task")

    #print('indices_k_unlearn :{}, total:{}'.format(indices_k_unlearn, len(indices_k_unlearn)))
    print('{} indices_k_unlearn total:{}'.format(task, len(indices_k_unlearn)))

    copy_train_labels = train_labels.clone()
    copy_train_labels[indices_k_unlearn] = -10
    indices_retain_data = (copy_train_labels != -10).nonzero(as_tuple=False).view(-1)
    unlearn_dataset = Subset(trainset, indices_k_unlearn.view(-1,))
    unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=unlearn_batch_size, 
                                            shuffle=False, num_workers=4, pin_memory=True)
    retain_dataset = Subset(trainset, indices_retain_data.view(-1,))
    retain_loader = torch.utils.data.DataLoader(retain_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=4, pin_memory=True)
    teacher_loader = create_teacher_dataloader(trainset,indices_k_unlearn,batch_size)

    num_retain_samples = int(len(retain_dataset) * retain_percentage)
    num_retain_samples = max(num_retain_samples, batch_size)
    fixed_retain_indices = np.random.choice(
        len(retain_dataset),
        size=num_retain_samples,
        replace=False
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(fixed_retain_indices),
        drop_last=True,
        num_workers=4
    )


    weights = torch.ones(len(unlearn_dataset), dtype=torch.double)
    unlearn_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_retain_samples, 
        replacement=True
    )
    iiae_unlearn_loader = DataLoader(
        unlearn_dataset,
        batch_size=batch_size,
        sampler=unlearn_sampler,
        drop_last=True,
        num_workers=4
    )

    #select backbone
    if backbone == "resnet10":
        model = resnet10(num_classes=num_classes)
        init_model = resnet10(num_classes=num_classes)
        hidden_dim = 128
        latent_dim = 64
        vae_epochs = 1000
    elif backbone=="resnet50":
        model = resnet50(num_classes=num_classes)
        init_model = resnet50(num_classes=num_classes)
        hidden_dim = 128
        latent_dim = 64
        vae_epochs = 1000
    else: 
        raise ValueError("Invalid backbone")
    
    model = model.to(device)

    #load pretrained weights 
    pre_trained_model_path = "models/{}/pre_trained_models/{}_{}_{}.pt".format(dataset,dataset,task,backbone)
    model.load_state_dict(torch.load(pre_trained_model_path, weights_only=False))
    

    if method == "disentangle_mu": 
        iiae_epochs = 30
        train_iiae = True
        if task == "random" :
            index = unlearn_k
            dataset_type = f"{dataset}_random"
        else :  # sub_class
            index = s_cls
            dataset_type = f"{dataset}_subclass"
        clas = f"{backbone}_{task}_{dataset}_{index}_{retain_percentage}"

        if backbone == "resnet10":
            input_dim = 512
        elif backbone == "resnet50":
            input_dim = 2048
        
        u_acc, r_acc, t_acc, train_acc, mia = disentangle_mu(model, train_loader, retain_loader, iiae_unlearn_loader, 
                                                                iiae_epochs, device, train_iiae, test_loader, clas, batch_size, input_dim=input_dim, dataset_type=dataset_type)

    else: 
        raise ValueError("Invalid method")
    

    generate_baseline_report(experiment_name, report_path, method, dataset, model, batch_size, selection_rate, u_acc, t_acc, r_acc, train_acc, mia)

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Machine Unlearning")
    parser.add_argument('--experiment_name', type=str, default="experiment name")
    parser.add_argument('--task', type=str,default='random')
    parser.add_argument('--method', type=str, default="neggrad")
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--backbone', type=str, default='resnet10')
    parser.add_argument('--unlearn_k', type=int, default = 64) #unleran_ks = [4, 8, 16, 32, 64, 128, 256]
    parser.add_argument('--f_cls', type=int, default=0) #CIFAR10:[0,5,9], CIFAR100:[0,50,99] 
    parser.add_argument('--s_cls', type=int, default=69) #CIFAR20:[69,71,2]
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--selection_rate', type=float, default=0.05)
    parser.add_argument('--device', type=str,default='0')
    parser.add_argument('--num_epochs', type=int, default=100)   
    parser.add_argument('--retain_percentage', type=float, default=1.0)   
    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    task = args.task
    method = args.method
    dataset = args.dataset
    backbone = args.backbone
    unlearn_k = args.unlearn_k
    f_cls = args.f_cls
    s_cls = args.s_cls
    batch_size = args.batch_size
    selection_rate = args.selection_rate
    device = args.device
    num_epochs = args.num_epochs
    retain_percentage = args.retain_percentage

    main(experiment_name, task, method, dataset, backbone, unlearn_k, f_cls, s_cls, batch_size, selection_rate, device, num_epochs, retain_percentage)

    end_time = time.time()  
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")