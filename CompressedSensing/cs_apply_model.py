import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import sys
import os

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


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualRefiner()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def refine_image(model, device, cs_image_path, output_path):
    img = Image.open(cs_image_path).convert('L')
    original_size = img.size
    
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        refined_tensor = model(img_tensor)
        refined_tensor = torch.clamp(refined_tensor, 0, 1)
    
    refined_img = ToPILImage()(refined_tensor.squeeze(0).cpu())
    refined_img = refined_img.resize(original_size)
    refined_img.save(output_path)
    
    return refined_img


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python cs_apply_model.py <modelo.pth> <imagem_cs.jpg> [saida.jpg]")
        print("Exemplo: python cs_apply_model.py cs_refiner_model.pth img_reconstructed_l1.jpg img_refined.jpg")
        exit(1)
    
    model_path = sys.argv[1]
    cs_image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else "img_refined.jpg"
    
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo {model_path} nao encontrado!")
        print("Treine primeiro com: python cs_train_dataset.py")
        exit(1)
    
    if not os.path.exists(cs_image_path):
        print(f"ERRO: Imagem {cs_image_path} nao encontrada!")
        exit(1)
    
    print(f"Carregando modelo: {model_path}")
    model, device = load_model(model_path)
    
    print(f"Refinando imagem: {cs_image_path}")
    refined_img = refine_image(model, device, cs_image_path, output_path)
    
    print(f"Imagem refinada salva em: {output_path}")


