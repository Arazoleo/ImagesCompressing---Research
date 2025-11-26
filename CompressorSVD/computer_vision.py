import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Include the entire previous SVDCompressor class here
class SVDCompressor:
    def __init__(self, rank_percentage=0.1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.rank_percentage = rank_percentage

    def load_image(self, image_path):
        # Load and convert image to grayscale
        img = Image.open(image_path).convert('L')
        transform = transforms.ToTensor()
        return transform(img).to(self.device)

    def compress_svd(self, tensor):
        # Perform SVD on the image tensor
        U, S, V = torch.svd(tensor.squeeze())
        
        # Determine the number of singular values to keep
        k = int(len(S) * self.rank_percentage)
        
        # Truncate singular values and reconstruct
        U_trunc = U[:, :k]
        S_trunc = S[:k]
        V_trunc = V[:, :k]
        
        compressed = U_trunc @ torch.diag(S_trunc) @ V_trunc.t()
        return compressed, (U_trunc, S_trunc, V_trunc)

    def compress_and_refine(self, image_path):
        # Main compression and refinement pipeline
        original = self.load_image(image_path)
        
        # Ensure original is a 2D tensor
        original = original.squeeze(0)
        
        # Compress with SVD
        compressed, svd_components = self.compress_svd(original.unsqueeze(0))
        compressed = compressed.squeeze(0)
        
        # Create a new tensor that requires gradients
        compressed_opt = compressed.clone().detach().requires_grad_(True)
        
        # Refine compression
        refined = self.refine_compression(
            compressed_opt.unsqueeze(0), 
            original.unsqueeze(0)
        )
        
        return {
            'original': original,
            'compressed': compressed,
            'refined': refined.squeeze(0),
            'svd_components': svd_components
        }

    def refine_compression(self, compressed_tensor, original_tensor, iterations=10):
        # Ensure compressed_tensor requires gradients
        compressed_tensor = compressed_tensor.clone().detach().requires_grad_(True)
        
        # Create optimizer with the tensor that requires gradients
        optimizer = torch.optim.Adam([compressed_tensor], lr=0.01)
        criterion = nn.MSELoss()
        
        for _ in range(iterations):
            optimizer.zero_grad()
            loss = criterion(compressed_tensor, original_tensor)
            loss.backward()
            optimizer.step()
        
        return compressed_tensor

    def save_image(self, tensor, path):
        # Convert tensor to PIL Image
        img = transforms.ToPILImage()(tensor.cpu().squeeze())
        img.save(path)

# Streamlit Dashboard
def create_dashboard():
    st.title('SVD Image Compression Dashboard')
    
    # Sidebar for configuration
    st.sidebar.header('Compression Settings')
    rank_percentage = st.sidebar.slider(
        'Rank Percentage', 
        min_value=0.01, 
        max_value=1.0, 
        value=0.2, 
        step=0.01
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Initialize compressor
        compressor = SVDCompressor(rank_percentage=rank_percentage)
        
        # Perform compression
        results = compressor.compress_and_refine(uploaded_file)
        
        # Normalize images to [0, 1] range
        def normalize_image(img):
            img_np = img.detach().cpu().numpy().squeeze()
            return (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader('Original Image')
            st.image(normalize_image(results['original']), use_column_width=True)
        
        with col2:
            st.subheader('Compressed')
            st.image(normalize_image(results['compressed']), use_column_width=True)
        
        with col3:
            st.subheader('Refined Image')
            st.image(normalize_image(results['refined']), use_column_width=True)
        
        # Rest of the code remains the same...
        
        # Rest of the code remains the same...
        
        # Singular values visualization
        st.header('Singular Value Components')
        U, S, V = results['svd_components']
        
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader('Singular Values')
            fig, ax = plt.subplots()
            ax.plot(S.cpu().numpy(), marker='o')
            ax.set_title('Singular Value Distribution')
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            st.pyplot(fig)
        
        with col5:
            st.subheader('Compression Details')
            st.write(f"Rank Percentage: {rank_percentage}")
            st.write(f"Singular Values Kept: {len(S)}")
            st.write(f"Original Shape: {results['original'].shape}")
            st.write(f"Compressed Shape: {results['compressed'].shape}")

# Run the dashboard
if __name__ == '__main__':
    create_dashboard()
'''

This dashboard integrates directly with the previous SVD compression code, adding Streamlit visualization. Key features:
- Side-by-side image comparison
- Singular value distribution plot
- Compression details
- Interactive rank percentage slider

Run with:
```bash
streamlit run svd_compression_dashboard.py
'''