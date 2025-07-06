import torch 
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
import copy
import numpy as np 
import random
from sklearn.metrics import accuracy_score
import os
import torch.nn as nn
import torch.optim as optim


def to_list(tensor_list):
    tensor_list = [tensor.cpu() for tensor in tensor_list]
    temp = [] 
    for i in tensor_list:
        for item in i:
            temp.append(item.item())
    return temp 

def get_accuracy(answers, predictions): 
    total_acc =  accuracy_score(answers,predictions)
    print(total_acc)

    return total_acc 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    data_loader = torch.utils.data.DataLoader(
        data_loader.dataset, batch_size=1, shuffle=False
    )
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data,target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model):
    model = copy.deepcopy(model)
    retain_prob = collect_prob(retain_loader, model)
    forget_prob = collect_prob(forget_loader, model)
    test_prob = collect_prob(test_loader, model)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model):
    model = copy.deepcopy(model)
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model
    )
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial",random_state=42
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    accuracy = (results == Y_f).mean()
    return accuracy

def get_featuremap_extractor(model, layer_name="conv5_x.0.residual_function.3"):
    feat_dict = {}
    def hook_fn(module, input, output):
        feat_dict["feat"] = output.detach()
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(hook_fn)
            break
    else:
        raise ValueError(f"Layer {layer_name} not found")
    def extractor(x):
        feat_dict.clear()
        _ = model(x)
        fmap = feat_dict["feat"]
        return fmap.view(fmap.size(0), -1)
    return extractor, hook

def extract_conv_features(loader, model, extractor, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            f = extractor(x)
            feats.append(f)
            labels.append(y)
    X = torch.cat(feats, dim=0)
    Y = torch.cat(labels, dim=0)
    return X, Y

class MIA_Classifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(1)

def get_membership_attack_map(retain_loader, forget_loader, test_loader, model,
                                feature_layer="conv5_x.0.residual_function.3",device=None):
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    extractor, hook = get_featuremap_extractor(model, feature_layer)

    X_retain, Y_retain = extract_conv_features(retain_loader, model, extractor, device)
    X_test,   Y_test   = extract_conv_features(test_loader,   model, extractor, device)
    X_forget, Y_forget = extract_conv_features(forget_loader, model, extractor, device)

    Y_retain = torch.ones_like(Y_retain, dtype=torch.float, device=device)
    Y_test   = torch.zeros_like(Y_test,   dtype=torch.float, device=device)
    Y_forget = torch.ones_like(Y_forget, dtype=torch.float, device=device)

    X_train = torch.cat([X_retain, X_test], dim=0)   
    Y_train = torch.cat([Y_retain, Y_test], dim=0)  

    in_dim = X_train.size(1)
    clf = MIA_Classifier(in_dim).to(device)
    optimizer = optim.SGD(clf.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    clf.train()
    for epoch in range(5):
        perm = torch.randperm(X_train.size(0), device=device)
        for i in range(0, perm.size(0), 128):
            idx = perm[i:i+128]
            xb = X_train[idx]
            yb = Y_train[idx]
            optimizer.zero_grad()
            loss = criterion(clf(xb), yb)
            loss.backward()
            optimizer.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(X_forget)          
        preds  = (logits > 0).float()   
        acc    = (preds == Y_forget).float().mean().item()

    hook.remove()
    return acc

def generate_baseline_report(experiment_name, path, method, dataset, model, batch_size, selection_rate, unlearn_acc, test_acc, retain_acc, train_acc, mia): 
    print("generating report")
    directory = os.path.dirname(path)
    # If the directory exists and is not empty, check for its existence and create if missing
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w') as file: 
        file.write("CONFIGURATIONS\n")
        file.write("{}\n".format(experiment_name))
        file.write("Method:{}\n".format(method))
        file.write("Dataset:{}\n".format(dataset))
        file.write("Backbone_model:{}\n".format(model.__class__.__name__))
        file.write('Batch_size:{}\n'.format(batch_size))

        file.write('Hyperparameters\n')
        file.write('selection rate:{}\n'.format(selection_rate))

        file.write('PERFORMANCE\n')
        file.write('unlearn acc:{}\n'.format(unlearn_acc))
        file.write('retain acc:{}\n'.format(retain_acc))
        file.write('test acc:{}\n'.format(test_acc))
        file.write('train acc:{}\n'.format(train_acc))
        file.write('mia:{}\n'.format(str(mia)))

def evaluate(model, loader, device):
    total_samples = 0 
    total_correct = 0 

    with torch.no_grad(): 
        model.eval() 

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            output = model(images)
            _,pred = torch.max(output.data,1)
            total_correct += (pred==labels).sum().item()
            total_samples += (labels.size(0))
    acc = total_correct/total_samples
    
    return acc 