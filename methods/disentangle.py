import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from itertools import cycle
import csv
import os

from data import *
from utils import *
from train import *


################################################################
class IIAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(IIAE, self).__init__()

        self.encoder_x = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  
        )
        self.encoder_y = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  
        )

        self.encoder_s = nn.Sequential(
            nn.Linear(input_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, latent_dim * 2)  
        )

        self.decoder_x = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.decoder_y = nn.Sequential(
            nn.Linear(latent_dim * 2, 512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, encoder, x):
        mean_logvar = encoder(x)
        mean, logvar = mean_logvar.chunk(2, dim=-1)  
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std  

    def forward(self, x, y):
        zx_mean, zx_logvar = self.encode(self.encoder_x, x)
        zy_mean, zy_logvar = self.encode(self.encoder_y, y)

        zs_mean, zs_logvar = self.encode(self.encoder_s, torch.cat([x, y], dim=1))

        zx = self.reparameterize(zx_mean, zx_logvar)
        zy = self.reparameterize(zy_mean, zy_logvar)
        zs = self.reparameterize(zs_mean, zs_logvar)

        x_recon = self.decoder_x(torch.cat([zx, zs], dim=1))
        y_recon = self.decoder_y(torch.cat([zy, zs], dim=1))

        return x_recon, y_recon, zx_mean, zx_logvar, zy_mean, zy_logvar, zs_mean, zs_logvar
    
    def cross_decode(self, x, y):
        zx_mean, zx_logvar = self.encode(self.encoder_x, x)
        zy_mean, zy_logvar = self.encode(self.encoder_y, y)

        zs_mean, zs_logvar = self.encode(self.encoder_s, torch.cat([x, y], dim=1))

        zx = self.reparameterize(zx_mean, zx_logvar)
        zy = self.reparameterize(zy_mean, zy_logvar)
        zs = self.reparameterize(zs_mean, zs_logvar)

        x_recon = self.decoder_x(torch.cat([zx, zs], dim=1))
        y_recon = self.decoder_y(torch.cat([zy, zs], dim=1))

        noise = torch.randn_like(zx)
        noise_s_recon_x = self.decoder_x(torch.cat([noise, zs], dim=1))
        noise_s_recon_y = self.decoder_y(torch.cat([noise, zs], dim=1))

        noise = torch.randn_like(zs)
        x_noise_recon = self.decoder_x(torch.cat([zx, noise], dim=1))
        y_noise_recon = self.decoder_y(torch.cat([zy, noise], dim=1))

        return x_recon, y_recon, x_noise_recon, y_noise_recon, noise_s_recon_x, noise_s_recon_y

################################################################
def gaussian_kl(mu_q, logvar_q, mu_p=None, logvar_p=None, eps=1e-6):
    if mu_p is None:
        mu_p = torch.zeros_like(mu_q)
    if logvar_p is None:
        logvar_p = torch.zeros_like(logvar_q)

    # clamp again just in case
    logvar_q = logvar_q.clamp(-8, 8)
    logvar_p = logvar_p.clamp(-8, 8)


    kl = 0.5 * torch.mean(
        logvar_p - logvar_q +
        (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1
    )
    return kl

################################################################
def train_IIAE(iiae, unlearn_loader, retain_loader, model, iiae_epochs, device, index):
    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    else:
        model.fc = nn.Identity()

    lambda_kl = 10
    lambda_info = 0.001
    optimizer = optim.Adam(iiae.parameters(), lr=1e-4)

    unlearn_iter = cycle(unlearn_loader)
    retain_iter = cycle(retain_loader)
    
    num_iterations = max(len(retain_loader), len(unlearn_loader))

    final_losses = []
    elbo_losses = []  
    info_losses = []

    for epoch in range(iiae_epochs):
        iiae.train()
        total_loss_epoch = 0
        total_recon_loss_epoch = 0
        total_kl_loss_epoch = 0
        total_info_loss_epoch = 0

        for iter_idx in range(num_iterations):
            r_imgs, _ = next(retain_iter)
            u_imgs, _ = next(unlearn_iter)
            
            r_imgs = r_imgs.to(device)
            u_imgs = u_imgs.to(device)

            with torch.no_grad():
                retain_features = model(r_imgs)
                unlearn_features = model(u_imgs)

            x_recon, y_recon, zx_mean, zx_logvar, zy_mean, zy_logvar, zs_mean, zs_logvar = iiae(unlearn_features, retain_features)

            recon_x = F.mse_loss(x_recon, unlearn_features, reduction='sum') 
            recon_y = F.mse_loss(y_recon, retain_features, reduction='sum') 

            def kl_term(mu, lv):
                return -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            kl_x = kl_term(zx_mean, zx_logvar)
            kl_y = kl_term(zy_mean, zy_logvar)
            kl_s = kl_term(zs_mean, zs_logvar)

            kl_zs_zx = gaussian_kl(zs_mean, zs_logvar, zx_mean, zx_logvar)
            kl_zs_zy = gaussian_kl(zs_mean, zs_logvar, zy_mean, zy_logvar)

            fused_m = 0.5*(zx_mean + zy_mean)
            fused_lv = 0.5*(zx_logvar + zy_logvar)
            kl_zs_fused = gaussian_kl(fused_m, fused_lv, zs_mean, zs_logvar)

            loss_info = -(kl_zs_zx + kl_zs_zy) + 2 * kl_zs_fused
            loss_recon = recon_x + recon_y
            loss_kl = kl_x + kl_y + kl_s
            
            final_loss = loss_recon + (lambda_kl * loss_kl) + (lambda_info * loss_info)
            
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            total_loss_epoch += final_loss.item()
            total_recon_loss_epoch += loss_recon.item()
            total_kl_loss_epoch += (lambda_kl * loss_kl).item()
            total_info_loss_epoch += (lambda_info * loss_info).item()

        # 에포크 평균 손실 계산
        avg_loss = total_loss_epoch / num_iterations
        avg_elbo_loss = (total_recon_loss_epoch + total_kl_loss_epoch) / num_iterations
        avg_info_loss = total_info_loss_epoch / num_iterations

        # Loss 기록
        final_losses.append(avg_loss)
        elbo_losses.append(avg_elbo_loss)
        info_losses.append(avg_info_loss)

        if epoch % 10 == 0 or epoch == iiae_epochs - 1:
            print(
                f"IIAE Epoch [{epoch+1}/{iiae_epochs}]"
                f"\n  [Total Loss]     : {avg_loss:.2f}"
                f"\n  [Reconstruction] : total={total_recon_loss_epoch/num_iterations:.2f} (x={recon_x.item():.2f}, y={recon_y.item():.2f})"
                f"\n  [KL Total]       : {total_kl_loss_epoch/num_iterations:.2f} (kl_x={kl_x.item():.2f}, kl_y={kl_y.item():.2f}, kl_s={kl_s.item():.2f})"
                f"\n  [Info Loss]      : {avg_info_loss:.2f} (kl_zs_zx={kl_zs_zx.item():.2f}, kl_zs_zy={kl_zs_zy.item():.2f}, kl_zs_fused={kl_zs_fused.item():.2f})"
            )

        if torch.isnan(torch.tensor(avg_loss)): 
            print("NaN loss detected. Stopping training.")
            break
    
    os.makedirs('./iiae/models', exist_ok=True)
    torch.save(iiae.state_dict(), f"./iiae/models/iiae_model_{index}.pt")
    print("IIAE model saved successfully.")

################################################################
def disentangle_mu(model, train_loader, retain_loader, unlearn_loader, iiae_epochs, 
                    device, train_iiae, val_loader, index, batch_size, input_dim=512, 
                    dataset_type="TinyImageNet_random"):
    
    model.eval()

    latent_dim = 32

    for param in model.parameters():
        param.requires_grad = False

    u_acc = evaluate(model, unlearn_loader, device)
    r_acc = evaluate(model, retain_loader, device)
    t_acc = evaluate(model, val_loader, device)
    print(f"Init Evaluation - Unlearn: {u_acc:4f}, Retain: {r_acc:4f}, Val: {t_acc:4f}")

    classifier = model.classifier if hasattr(model, 'classifier') else model.fc

    if hasattr(model, 'classifier'):
        model.classifier = nn.Identity()
    else:
        model.fc = nn.Identity()
    
    with torch.no_grad():
        sample_input = next(iter(unlearn_loader))[0][:1].to(device)
        sample_features = model(sample_input)
        actual_input_dim = sample_features.shape[1]
        print(f"Detected actual feature dimension: {actual_input_dim}")


    iiae = IIAE(actual_input_dim, latent_dim).to(device)

    if train_iiae: 
        train_IIAE(iiae, unlearn_loader, retain_loader, model, iiae_epochs, device, index)
    else: 
        try:    
            iiae.load_state_dict(torch.load(f"./iiae/models/iiae_model_{index}.pt", map_location=device, weights_only=True))
            print("IIAE model loaded successfully.")
        except Exception as e:
            print(e)
            train_IIAE(iiae, unlearn_loader, retain_loader, model, iiae_epochs, device, index)
    # Freeze
    for param in iiae.parameters():
        param.requires_grad = False
    iiae.eval() 
    
    current_model = copy.deepcopy(model).to(device)
    if hasattr(current_model, 'classifier'):
        current_model.classifier = nn.Identity()
    else:
        current_model.fc = nn.Identity()

    for param in current_model.parameters():
        param.requires_grad = True

    if dataset_type == "TinyImageNet_random" :
        fine_tune_epochs = 5
    elif dataset_type == "CIFAR20_subclass" :
        fine_tune_epochs = 3
    else :
        raise ValueError("Invalid dataset type")
    
    optimizer = optim.Adam(current_model.parameters(), lr=1e-6)
    mse_loss = nn.MSELoss()
    
    results = []

    for epoch in range(fine_tune_epochs):
        retain_iter = cycle(retain_loader)
        unlearn_iter = cycle(unlearn_loader)

        for _ in range(len(retain_loader)):
            retain_data = next(retain_iter)
            unlearn_data = next(unlearn_iter)

            retain_imgs, _ = retain_data
            unlearn_imgs, _ = unlearn_data

            while retain_imgs.size(0) < batch_size:
                extra_imgs, extra_labels = next(retain_iter)
                retain_imgs = torch.cat([retain_imgs, extra_imgs], dim=0)
            retain_imgs = retain_imgs[:batch_size]

            while unlearn_imgs.size(0) < batch_size:
                extra_imgs, extra_labels = next(unlearn_iter)
                unlearn_imgs = torch.cat([unlearn_imgs, extra_imgs], dim=0)
            unlearn_imgs = unlearn_imgs[:batch_size]

            retain_imgs = retain_imgs.to(device)
            unlearn_imgs = unlearn_imgs.to(device)

            retain_features = current_model(retain_imgs)
            unlearn_features = current_model(unlearn_imgs)

            zx_mean, _ = iiae.encode(iiae.encoder_x, unlearn_features)
            zs_mean, _ = iiae.encode(iiae.encoder_s, torch.cat([unlearn_features, retain_features], dim=1))
            zy_mean, _ = iiae.encode(iiae.encoder_y, retain_features)
            
            with torch.no_grad():
                init_retain_features = model(retain_imgs)
                init_unlearn_features = model(unlearn_imgs)

            init_zs_mean, _ = iiae.encode(iiae.encoder_s, torch.cat([init_unlearn_features, init_retain_features], dim=1))
            init_zy_mean, _ = iiae.encode(iiae.encoder_y, init_retain_features)

            zero = torch.zeros_like(init_zs_mean)
            loss_alignment = mse_loss(zx_mean, zero)
            loss_consistency = mse_loss(zs_mean, init_zs_mean) + mse_loss(zy_mean, init_zy_mean)
            loss =  1/2 * loss_alignment + loss_consistency
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if hasattr(current_model, 'classifier'):
            current_model.classifier = classifier
        else:
            current_model.fc = classifier
        current_model.eval()

        u_acc = evaluate(current_model, unlearn_loader, device)
        r_acc = evaluate(current_model, retain_loader, device)
        t_acc = evaluate(current_model, val_loader, device)
        train_acc = evaluate(current_model, train_loader, device)
        mia = get_membership_attack_prob(retain_loader, unlearn_loader, val_loader, current_model)
        
        print(f"Evaluation epoch {epoch + 1} - Unlearn: {u_acc:.4f}, Retain: {r_acc:.4f}, Val: {t_acc:.4f}")
        results.append([epoch + 1, u_acc, r_acc, train_acc, t_acc, mia])
        
        if hasattr(current_model, 'classifier'):
            current_model.classifier = nn.Identity()
        else:
            current_model.fc = nn.Identity()
        current_model.train()

    os.makedirs("./log_disen/", exist_ok=True)

    csv_filename = f"./log_disen/results_{index}_{dataset_type}.csv"

    with open(csv_filename, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["epoch", "u_acc", "r_acc", "train_acc", "t_acc", "mia"])
        csv_writer.writerows(results)


    return u_acc, r_acc, t_acc, train_acc, mia
