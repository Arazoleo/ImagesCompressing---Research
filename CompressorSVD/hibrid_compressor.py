import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image

# --- Carregar a Imagem Pós-SVD ---
def load_pos_svd_image(image_path, target_size=(424, 424)):
    image = Image.open(image_path).convert("L")
    image = image.resize(target_size)
    return image

# --- Definição do Autoencoder ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Treinar o Autoencoder ---
def train_autoencoder(image, num_epochs=5000, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Converter imagem para tensor normalizado
    img_tensor = ToTensor()(image).unsqueeze(0).to(device)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        output = model(img_tensor)
        loss = criterion(output, img_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# --- Refinar Imagem com Autoencoder Treinado ---
def refine_image(model, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        refined_tensor = model(image_tensor).squeeze(0).cpu()

    refined_image = refined_tensor.numpy().squeeze() * 255
    return refined_image.astype(np.uint8)

# --- Executar o Processo ---
compressed_image = load_pos_svd_image("img_compressed.jpeg")

# Treinar o autoencoder
model = train_autoencoder(compressed_image)

# Refinar a imagem comprimida
refined_image = refine_image(model, compressed_image)

# --- Exibir Imagens ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem Comprimida pela SVD")
plt.imshow(compressed_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Imagem Refinada pelo Autoencoder")
plt.imshow(refined_image, cmap="gray")
plt.axis("off")

plt.show()
