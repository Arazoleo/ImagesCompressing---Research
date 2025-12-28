import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Resize
import os
import glob
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class ResidualRefiner(nn.Module):
    def __init__(self):
        super(ResidualRefiner, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
        )
        
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.initial(x)
        feat = self.residual_blocks(feat)
        residual = self.final(feat)
        return x + residual


class CSDataset(Dataset):
    def __init__(self, original_dir, cs_dir, size=(256, 256)):
        self.original_dir = original_dir
        self.cs_dir = cs_dir
        self.size = size
        
        original_files = sorted(glob.glob(os.path.join(original_dir, "*.jpg")) + 
                               glob.glob(os.path.join(original_dir, "*.png")))
        cs_files = sorted(glob.glob(os.path.join(cs_dir, "*.jpg")) + 
                         glob.glob(os.path.join(cs_dir, "*.png")))
        
        self.pairs = []
        for orig_file in original_files:
            filename = os.path.basename(orig_file)
            cs_file = os.path.join(cs_dir, filename)
            if os.path.exists(cs_file):
                self.pairs.append((orig_file, cs_file))
        
        print(f"Encontrados {len(self.pairs)} pares de imagens")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        orig_path, cs_path = self.pairs[idx]
        
        orig_img = Image.open(orig_path).convert('L')
        cs_img = Image.open(cs_path).convert('L')
        
        orig_img = orig_img.resize(self.size)
        cs_img = cs_img.resize(self.size)
        
        orig_tensor = ToTensor()(orig_img)
        cs_tensor = ToTensor()(cs_img)
        
        return cs_tensor, orig_tensor


def train_on_dataset(original_dir, cs_dir, model_save_path, num_epochs=50, batch_size=4, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    dataset = CSDataset(original_dir, cs_dir, size=(256, 256))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = ResidualRefiner().to(device)
    print(f"Parametros do modelo: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    losses = []
    best_loss = float('inf')
    
    print(f"Iniciando treinamento em {len(dataset)} imagens...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for cs_batch, orig_batch in pbar:
            cs_batch = cs_batch.to(device)
            orig_batch = orig_batch.to(device)
            
            output = model(cs_batch)
            loss = criterion(output, orig_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Melhor modelo salvo! Loss: {best_loss:.6f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}, Best: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"Treinamento concluido!")
    print(f"Modelo salvo em: {model_save_path}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig('training_loss_dataset.png', dpi=150)
    plt.show()
    
    return model


if __name__ == "__main__":
    original_dir = "dataset/original"
    cs_dir = "dataset/cs_reconstructed"
    model_path = "cs_refiner_model.pth"
    
    if not os.path.exists(original_dir):
        print(f"ERRO: Diretorio {original_dir} nao existe!")
        print("Crie a estrutura:")
        print("  dataset/original/     - imagens originais")
        print("  dataset/cs_reconstructed/ - imagens comprimidas pelo CS L1")
        exit(1)
    
    model = train_on_dataset(
        original_dir=original_dir,
        cs_dir=cs_dir,
        model_save_path=model_path,
        num_epochs=50,
        batch_size=4,
        lr=0.0001
    )


