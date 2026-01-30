import os
import urllib.request
from PIL import Image
import io

def download_set5():
    """
    Baixa Set5 - dataset pequeno (5 imagens) usado em super-resolução
    """
    urls = {
        "baby.png": "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5/image_SRF_2/baby_SRF_2_HR.png",
        "bird.png": "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5/image_SRF_2/bird_SRF_2_HR.png",
        "butterfly.png": "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5/image_SRF_2/butterfly_SRF_2_HR.png",
        "head.png": "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5/image_SRF_2/head_SRF_2_HR.png",
        "woman.png": "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set5/image_SRF_2/woman_SRF_2_HR.png",
    }
    
    original_dir = "dataset/original"
    os.makedirs(original_dir, exist_ok=True)
    
    print("Baixando Set5 dataset (5 imagens)...")
    
    for i, (name, url) in enumerate(urls.items(), 1):
        filename = name.replace('.png', '.jpg')
        filepath = os.path.join(original_dir, filename)
        
        try:
            print(f"[{i}/{len(urls)}] Baixando {filename}...")
            response = urllib.request.urlopen(url)
            img_data = response.read()
            
            img = Image.open(io.BytesIO(img_data)).convert('L')
            img.save(filepath, 'JPEG')
            print(f"  OK: {filename}")
            
        except Exception as e:
            print(f"  ERRO ao baixar {filename}: {e}")
    
    print(f"\nDataset Set5 preparado em {original_dir}")
    print(f"Total: {len(os.listdir(original_dir))} imagens")


def download_celeba_sample():
    """
    Baixa uma amostra do CelebA (rostos)
    Requer acesso ao dataset, então vamos usar Set5 como padrão
    """
    print("CelebA requer download manual.")
    print("Usando Set5 como alternativa...")
    download_set5()


if __name__ == "__main__":
    print("=== Download de Dataset Pronto ===\n")
    print("Baixando Set5 (5 imagens de teste)...")
    print("Este dataset e pequeno e rapido para testes.\n")
    
    download_set5()
    
    print("\n=== Proximos Passos ===")
    print("1. Gerar imagens CS: python generate_cs_images.py")
    print("   (Ou aplique CS L1 manualmente em cada imagem)")
    print("2. Treinar modelo: python cs_train_dataset.py")
    print("\nNota: Para mais imagens, voce pode:")
    print("- Baixar Set14: python download_dataset.py (opcao 1)")
    print("- Usar suas proprias imagens: python prepare_dataset.py <pasta>")




