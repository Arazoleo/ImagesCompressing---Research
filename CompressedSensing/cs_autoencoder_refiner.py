import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

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


def load_image(path):
    img = Image.open(path).convert('L')
    return img


def image_to_tensor(image):
    return ToTensor()(image).unsqueeze(0)


def tensor_to_image(tensor):
    tensor = torch.clamp(tensor, 0, 1)
    return ToPILImage()(tensor.squeeze(0).cpu())


def refine_and_compare(model, cs_tensor, original_tensor, output_path):
    model.eval()
    with torch.no_grad():
        refined_tensor = model(cs_tensor)
        refined_tensor = torch.clamp(refined_tensor, 0, 1)
    
    mse_before = nn.MSELoss()(cs_tensor, original_tensor).item()
    mse_after = nn.MSELoss()(refined_tensor, original_tensor).item()
    
    print(f"MSE antes do refinamento: {mse_before:.6f}")
    print(f"MSE apos refinamento:     {mse_after:.6f}")
    print(f"Melhoria: {((mse_before - mse_after) / mse_before * 100):.2f}%")
    
    cs_img = tensor_to_image(cs_tensor)
    refined_img = tensor_to_image(refined_tensor)
    original_img = tensor_to_image(original_tensor)
    
    refined_img.save(output_path)
    print(f"Imagem refinada salva em: {output_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cs_img, cmap='gray')
    axes[0].set_title(f'CS Reconstruida\nMSE: {mse_before:.6f}')
    axes[0].axis('off')
    
    axes[1].imshow(refined_img, cmap='gray')
    axes[1].set_title(f'Refinada (ResNet)\nMSE: {mse_after:.6f}')
    axes[1].axis('off')
    
    axes[2].imshow(original_img, cmap='gray')
    axes[2].set_title('Original')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('comparison_cs_refined.png', dpi=150)
    plt.show()
    
    return mse_before, mse_after


def train_refiner(original_path, cs_reconstructed_path, num_epochs=2000, lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    original_img = load_image(original_path)
    cs_img = load_image(cs_reconstructed_path)

    w, h = original_img.size
    new_w = (w // 4) * 4
    new_h = (h // 4) * 4

    original_img = original_img.resize((new_w, new_h))
    cs_img = cs_img.resize((new_w, new_h))

    print(f"Tamanho das imagens: {new_w} x {new_h}")

    original_tensor = image_to_tensor(original_img).to(device)
    cs_tensor = image_to_tensor(cs_img).to(device)

    model = ResidualRefiner().to(device)
    print(f"Parametros do modelo: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    best_loss = float('inf')

    print(f"Iniciando treinamento...")

    for epoch in range(num_epochs):
        model.train()

        output = model(cs_tensor)
        loss = criterion(output, original_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}, Best: {best_loss:.6f}")
            
    print(f"Treinamento concluido!")
    print(f"Loss final: {losses[-1]:.6f}")
    return model, losses, cs_tensor, original_tensor


if __name__ == "__main__":
    original_path = "img.jpg" 
    cs_path = "img_reconstructed_l1.jpg"
    refined_path = "img_refined_resnet.jpg"

    model, losses, cs_tensor, original_tensor = train_refiner(
        original_path,
        cs_path,
        num_epochs=2000,
        lr=0.0001
    )

    mse_before, mse_after = refine_and_compare(
        model,
        cs_tensor,
        original_tensor,
        refined_path
    )

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig('training_loss.png', dpi=150)
    plt.show()
