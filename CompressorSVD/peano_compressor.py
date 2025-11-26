import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve
from PIL import Image

# Função para mapear a imagem na curva de Hilbert
def image_to_peano(image, p=7):
    img_array = np.array(image)
    size = img_array.shape[0]
    max_size = 2**p
    assert size == max_size, f"Erro: A imagem deve ter tamanho {max_size}x{max_size} para p={p}."

    hilbert = HilbertCurve(p, 2)
    indices = np.array([hilbert.point_from_distance(i) for i in range(size * size)])
    indices = np.clip(indices, 0, size - 1)
    hilbert_sequence = img_array[indices[:, 1], indices[:, 0]].flatten()
    return hilbert_sequence, indices

# Função para reconstruir a imagem
def peano_to_image(hilbert_sequence, indices, shape):
    image_array = np.zeros(shape, dtype=np.uint8)
    image_array[indices[:, 1], indices[:, 0]] = hilbert_sequence
    return image_array

# Autoencoder
class HilbertAutoencoder(nn.Module):
    def __init__(self, input_size):
        super(HilbertAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Função de treinamento
def train_autoencoder(data, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(data / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
    
    model = HilbertAutoencoder(input_size=data.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        output = model(data)
        loss = criterion(output, data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model

# Interface do Streamlit
st.title("Compressão de Imagem com Curva de Hilbert e Autoencoder")

uploaded_file = st.file_uploader("Carregue uma imagem (JPG ou PNG)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((128, 128))
    st.image(image, caption="Imagem Original", use_column_width=True)
    
    hilbert_sequence, indices = image_to_peano(image, p=7)
    
    model = train_autoencoder(hilbert_sequence, num_epochs=50)
    
    with torch.no_grad():
        refined_sequence = model(torch.tensor(hilbert_sequence / 255.0, dtype=torch.float32).unsqueeze(0)).cpu().numpy().flatten()
        refined_sequence = (refined_sequence * 255).astype(np.uint8)
    
    reconstructed_image = peano_to_image(refined_sequence, indices, (128, 128))
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagem Original", use_column_width=True)
    with col2:
        st.image(reconstructed_image, caption="Imagem Reconstruída", use_column_width=True)
    
    st.write("### Comparação de Distribuição de Pixels")
    fig, ax = plt.subplots()
    ax.hist(hilbert_sequence, bins=50, alpha=0.5, label="Original", color='blue')
    ax.hist(refined_sequence, bins=50, alpha=0.5, label="Reconstruído", color='red')
    ax.legend()
    st.pyplot(fig)
