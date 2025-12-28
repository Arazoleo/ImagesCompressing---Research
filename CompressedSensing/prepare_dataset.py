import os
import shutil
import glob
from PIL import Image

def prepare_dataset_from_folder(source_folder, output_base="dataset"):
    """
    Prepara dataset a partir de uma pasta com imagens.
    Cria estrutura: dataset/original/ e dataset/cs_reconstructed/
    """
    
    original_dir = os.path.join(output_base, "original")
    cs_dir = os.path.join(output_base, "cs_reconstructed")
    
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(cs_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(source_folder, "*.jpg")) + \
                  glob.glob(os.path.join(source_folder, "*.png")) + \
                  glob.glob(os.path.join(source_folder, "*.jpeg"))
    
    print(f"Encontradas {len(image_files)} imagens em {source_folder}")
    print(f"Copiando para {original_dir}...")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        dest_path = os.path.join(original_dir, filename)
        
        try:
            img = Image.open(img_path)
            img = img.convert('L')
            img.save(dest_path)
        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")
            continue
    
    print(f"\nDataset preparado!")
    print(f"Imagens originais: {original_dir}")
    print(f"\nAgora voce precisa:")
    print(f"1. Aplicar CS L1 em todas as imagens de {original_dir}")
    print(f"2. Salvar as imagens comprimidas em {cs_dir}")
    print(f"3. Rodar: python cs_train_dataset.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python prepare_dataset.py <pasta_com_imagens>")
        print("Exemplo: python prepare_dataset.py ./imagens")
        exit(1)
    
    source_folder = sys.argv[1]
    
    if not os.path.exists(source_folder):
        print(f"ERRO: Pasta {source_folder} nao existe!")
        exit(1)
    
    prepare_dataset_from_folder(source_folder)


