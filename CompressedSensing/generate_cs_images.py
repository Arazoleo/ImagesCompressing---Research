import os
import glob
from PIL import Image
import subprocess
import sys

def generate_cs_for_dataset(original_dir, cs_dir, julia_script="compressive_sensing_l1.jl"):
    """
    Gera imagens CS L1 para todas as imagens do dataset.
    Requer que o Julia esteja instalado e o script CS L1 funcione.
    """
    
    os.makedirs(cs_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(original_dir, "*.jpg")) + \
                  glob.glob(os.path.join(original_dir, "*.png")) + \
                  glob.glob(os.path.join(original_dir, "*.jpeg"))
    
    print(f"Encontradas {len(image_files)} imagens")
    print(f"Gerando imagens CS L1...")
    
    julia_script_path = os.path.abspath(julia_script)
    
    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        output_path = os.path.join(cs_dir, filename)
        
        if os.path.exists(output_path):
            print(f"[{i}/{len(image_files)}] {filename} ja existe, pulando...")
            continue
        
        print(f"[{i}/{len(image_files)}] Processando {filename}...")
        
        try:
            img = Image.open(img_path).convert('L')
            img.save("temp_input.jpg")
            
            julia_code = f'''
using Images
using LinearAlgebra
using JuMP
using GLPK
using Random

include("{julia_script_path}")

img = "temp_input.jpg"
img_out = "{output_path}"
tax = 0.5

img_reconstructed, error = compress_image_cs_l1(img, tax, img_out)
println("Processado: {filename}, MSE: ", error)
'''
            
            with open("temp_julia_script.jl", "w") as f:
                f.write(julia_code)
            
            result = subprocess.run(
                ["julia", "temp_julia_script.jl"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  ERRO ao processar {filename}")
                print(f"  {result.stderr[:500]}")
            else:
                print(f"  OK: {output_path}")
        except Exception as e:
            print(f"  ERRO: {e}")
    
    if os.path.exists("temp_input.jpg"):
        os.remove("temp_input.jpg")
    if os.path.exists("temp_julia_script.jl"):
        os.remove("temp_julia_script.jl")
    
    num_cs = len([f for f in os.listdir(cs_dir) if f.endswith(('.jpg', '.png'))])
    print(f"\nConcluido! Imagens CS salvas: {num_cs}/{len(image_files)} em {cs_dir}")


if __name__ == "__main__":
    original_dir = "dataset/original"
    cs_dir = "dataset/cs_reconstructed"
    
    if not os.path.exists(original_dir):
        print(f"ERRO: Diretorio {original_dir} nao existe!")
        print("Execute primeiro: python prepare_dataset.py <pasta_imagens>")
        exit(1)
    
    generate_cs_for_dataset(original_dir, cs_dir)
