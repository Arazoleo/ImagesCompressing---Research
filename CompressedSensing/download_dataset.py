import os
import urllib.request
import zipfile
import tarfile
import shutil
from PIL import Image
import glob

def download_bsds300():
    """
    Baixa o dataset BSDS300 (Berkeley Segmentation Dataset)
    Dataset pequeno e popular para testes
    """
    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    filename = "BSDS300-images.tgz"
    
    print("Baixando BSDS300 dataset...")
    print("Isso pode demorar alguns minutos...")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print("Download concluido!")
        
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        
        os.remove(filename)
        
        original_dir = "dataset/original"
        os.makedirs(original_dir, exist_ok=True)
        
        train_dir = "BSDS300/images/train"
        test_dir = "BSDS300/images/test"
        
        for img_path in glob.glob(os.path.join(train_dir, "*.jpg")):
            shutil.copy(img_path, original_dir)
        
        for img_path in glob.glob(os.path.join(test_dir, "*.jpg")):
            shutil.copy(img_path, original_dir)
        
        print(f"Dataset preparado em {original_dir}")
        print(f"Total de imagens: {len(glob.glob(os.path.join(original_dir, '*.jpg')))}")
        
    except Exception as e:
        print(f"Erro ao baixar: {e}")
        print("Tentando alternativa...")


def download_set14():
    """
    Baixa o dataset Set14 (pequeno, popular para super-resolução)
    """
    urls = [
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_001_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_002_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_003_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_004_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_005_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_006_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_007_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_008_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_009_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_010_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_011_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_012_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_013_SRF_2_HR.png",
        "https://github.com/jbhuang0604/SelfExSR/raw/master/data/Set14/image_SRF_2/img_014_SRF_2_HR.png",
    ]
    
    original_dir = "dataset/original"
    os.makedirs(original_dir, exist_ok=True)
    
    print("Baixando Set14 dataset (14 imagens)...")
    
    for i, url in enumerate(urls, 1):
        filename = f"img_{i:03d}.png"
        filepath = os.path.join(original_dir, filename)
        
        try:
            print(f"[{i}/{len(urls)}] Baixando {filename}...")
            urllib.request.urlretrieve(url, filepath)
            
            img = Image.open(filepath).convert('L')
            img.save(filepath.replace('.png', '.jpg'), 'JPEG')
            os.remove(filepath)
            
        except Exception as e:
            print(f"  Erro ao baixar {filename}: {e}")
    
    print(f"Dataset Set14 preparado em {original_dir}")


def download_div2k_sample():
    """
    Baixa uma amostra do DIV2K (dataset grande de alta qualidade)
    Usa apenas algumas imagens para não demorar muito
    """
    base_url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    
    print("DIV2K e muito grande. Usando Set14 como alternativa...")
    download_set14()


def prepare_from_coco_sample():
    """
    Usa imagens do COCO dataset (se já tiver baixado)
    """
    print("COCO dataset requer download manual.")
    print("Use: download_set14() ou download_bsds300()")


if __name__ == "__main__":
    import sys
    
    print("=== Download de Datasets Prontos ===\n")
    print("Opcoes disponiveis:")
    print("1. Set14 (14 imagens) - Recomendado para testes")
    print("2. BSDS300 (300 imagens) - Dataset completo")
    print("3. Usar imagens locais")
    
    choice = input("\nEscolha (1/2/3): ").strip()
    
    if choice == "1":
        download_set14()
    elif choice == "2":
        download_bsds300()
    elif choice == "3":
        folder = input("Digite o caminho da pasta com imagens: ").strip()
        if os.path.exists(folder):
            from prepare_dataset import prepare_dataset_from_folder
            prepare_dataset_from_folder(folder)
        else:
            print("Pasta nao encontrada!")
    else:
        print("Opcao invalida. Baixando Set14 por padrao...")
        download_set14()
    
    print("\n=== Proximos Passos ===")
    print("1. Gerar imagens CS: python generate_cs_images.py")
    print("2. Treinar modelo: python cs_train_dataset.py")

