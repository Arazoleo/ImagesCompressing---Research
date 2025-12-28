import os
import sys
import subprocess
import time
from pathlib import Path

def print_step(step_num, total, message):
    print("\n" + "="*60)
    print(f"PASSO {step_num}/{total}: {message}")
    print("="*60)

def check_julia():
    """Verifica se Julia está instalado"""
    try:
        result = subprocess.run(["julia", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Julia encontrado")
            return True
    except:
        pass
    print("✗ Julia não encontrado! Instale Julia primeiro.")
    return False

def check_python_packages():
    """Verifica pacotes Python necessários"""
    try:
        import torch
        import PIL
        import tqdm
        print("✓ Pacotes Python OK")
        return True
    except ImportError as e:
        print(f"✗ Pacote faltando: {e}")
        print("  Instale com: pip install torch pillow tqdm")
        return False

def step1_download_dataset(dataset_choice="1"):
    """Passo 1: Baixar dataset"""
    print_step(1, 5, "BAIXANDO DATASET")
    
    if dataset_choice == "1":
        print("Usando Set5 (5 imagens - rápido para testes)")
        from download_dataset_simple import download_set5
        download_set5()
    elif dataset_choice == "2":
        print("Usando Set14 (14 imagens)")
        from download_dataset import download_set14
        download_set14()
    else:
        print("Opção inválida, usando Set5...")
        from download_dataset_simple import download_set5
        download_set5()
    
    if not os.path.exists("dataset/original"):
        print("ERRO: Dataset não foi criado!")
        return False
    
    num_images = len([f for f in os.listdir("dataset/original") if f.endswith(('.jpg', '.png'))])
    print(f"✓ Dataset preparado: {num_images} imagens")
    return True

def step2_generate_cs_images():
    """Passo 2: Gerar imagens CS L1"""
    print_step(2, 5, "GERANDO IMAGENS CS L1")
    
    original_dir = "dataset/original"
    cs_dir = "dataset/cs_reconstructed"
    
    os.makedirs(cs_dir, exist_ok=True)
    
    import glob
    from PIL import Image
    
    image_files = glob.glob(os.path.join(original_dir, "*.jpg")) + \
                  glob.glob(os.path.join(original_dir, "*.png"))
    
    print(f"Processando {len(image_files)} imagens...")
    print("Isso pode demorar (CS L1 é lento)...")
    
    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(cs_dir, f"{base_name}.jpg")
        
        if os.path.exists(output_path):
            print(f"[{i}/{len(image_files)}] {filename} já existe, pulando...")
            continue
        
        print(f"[{i}/{len(image_files)}] Processando {filename}...")
        
        julia_script_path = os.path.abspath("compressive_sensing_l1.jl")
        
        julia_code = f'''
using Images
using LinearAlgebra
using JuMP
using GLPK
using Random

include("{julia_script_path}")

img = "{img_path}"
img_out = "{output_path}"
tax = 0.5

try
    img_reconstructed, error = compress_image_cs_l1(img, tax, img_out)
    println("OK: ", "{filename}")
catch e
    println("ERRO: ", e)
end
'''
        
        with open("temp_cs_batch.jl", "w") as f:
            f.write(julia_code)
        
        try:
            result = subprocess.run(
                ["julia", "temp_cs_batch.jl"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  ✓ {filename}")
            else:
                print(f"  ✗ Erro em {filename}")
                print(f"    {result.stderr[:200]}")
        except Exception as e:
            print(f"  ✗ Erro: {e}")
    
    if os.path.exists("temp_cs_batch.jl"):
        os.remove("temp_cs_batch.jl")
    
    num_cs = len([f for f in os.listdir(cs_dir) if f.endswith('.jpg')])
    print(f"✓ Imagens CS geradas: {num_cs}/{len(image_files)}")
    return num_cs > 0

def step3_train_model(num_epochs=50, batch_size=4):
    """Passo 3: Treinar modelo"""
    print_step(3, 5, "TREINANDO MODELO IA")
    
    if not os.path.exists("dataset/cs_reconstructed"):
        print("ERRO: Imagens CS não encontradas!")
        return False
    
    print(f"Parâmetros: {num_epochs} épocas, batch_size={batch_size}")
    print("Isso pode demorar vários minutos...")
    
    try:
        from cs_train_dataset import train_on_dataset
        
        model = train_on_dataset(
            original_dir="dataset/original",
            cs_dir="dataset/cs_reconstructed",
            model_save_path="cs_refiner_model.pth",
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=0.0001
        )
        
        if os.path.exists("cs_refiner_model.pth"):
            print("✓ Modelo treinado e salvo!")
            return True
        else:
            print("✗ Modelo não foi salvo!")
            return False
    
    except Exception as e:
        print(f"✗ Erro no treinamento: {e}")
        import traceback
        traceback.print_exc()
        return False

def step4_test_model():
    """Passo 4: Testar modelo"""
    print_step(4, 5, "TESTANDO MODELO")
    
    if not os.path.exists("cs_refiner_model.pth"):
        print("✗ Modelo não encontrado!")
        return False
    
    import glob
    test_images = glob.glob("dataset/cs_reconstructed/*.jpg")[:3]
    
    if not test_images:
        print("✗ Nenhuma imagem CS para testar!")
        return False
    
    print(f"Testando em {len(test_images)} imagens...")
    
    try:
        from cs_apply_model import load_model, refine_image
        
        model, device = load_model("cs_refiner_model.pth")
        
        for cs_img in test_images:
            filename = os.path.basename(cs_img)
            output = f"test_{filename}"
            
            print(f"  Processando {filename}...")
            refine_image(model, device, cs_img, output)
            print(f"  ✓ {output}")
        
        print("✓ Testes concluídos!")
        return True
    
    except Exception as e:
        print(f"✗ Erro nos testes: {e}")
        return False

def step5_summary():
    """Passo 5: Resumo"""
    print_step(5, 5, "RESUMO FINAL")
    
    print("\n✓ PIPELINE COMPLETO CONCLUÍDO!\n")
    
    print("Arquivos gerados:")
    if os.path.exists("cs_refiner_model.pth"):
        size = os.path.getsize("cs_refiner_model.pth") / (1024*1024)
        print(f"  ✓ cs_refiner_model.pth ({size:.2f} MB)")
    
    test_files = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".jpg")]
    if test_files:
        print(f"  ✓ {len(test_files)} imagens de teste")
    
    print("\nComo usar em novas imagens:")
    print("  python cs_pipeline.py imagem.jpg resultado.jpg")
    print("\nOu aplicar apenas o refinador:")
    print("  python cs_apply_model.py cs_refiner_model.pth img_cs.jpg resultado.jpg")

def main():
    print("\n" + "="*60)
    print("PIPELINE COMPLETO: Dataset → CS L1 → Treino → Teste")
    print("="*60)
    
    if not check_julia():
        return
    
    if not check_python_packages():
        return
    
    print("\nEscolha o dataset:")
    print("  1. Set5 (5 imagens) - Rápido para testes")
    print("  2. Set14 (14 imagens) - Melhor qualidade")
    
    dataset_choice = input("\nEscolha (1 ou 2, padrão=1): ").strip() or "1"
    
    print("\nParâmetros de treinamento:")
    num_epochs = input("Número de épocas (padrão=50): ").strip()
    num_epochs = int(num_epochs) if num_epochs else 50
    
    batch_size = input("Batch size (padrão=4): ").strip()
    batch_size = int(batch_size) if batch_size else 4
    
    start_time = time.time()
    
    try:
        if not step1_download_dataset(dataset_choice):
            print("\n✗ Falha no passo 1")
            return
        
        if not step2_generate_cs_images():
            print("\n✗ Falha no passo 2")
            return
        
        if not step3_train_model(num_epochs, batch_size):
            print("\n✗ Falha no passo 3")
            return
        
        step4_test_model()
        
        step5_summary()
        
        elapsed = time.time() - start_time
        print(f"\n⏱ Tempo total: {elapsed/60:.1f} minutos")
        
    except KeyboardInterrupt:
        print("\n\n✗ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


