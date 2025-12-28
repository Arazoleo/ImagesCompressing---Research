import os
import sys
import subprocess
import torch
from PIL import Image

def run_cs_l1(input_image, output_cs, tax=0.5):
    """
    Aplica CS L1 na imagem usando Julia
    """
    print(f"1. Aplicando CS L1 (taxa: {tax*100}%)...")
    
    julia_script_path = os.path.abspath("compressive_sensing_l1.jl")
    
    julia_code = f'''
using Images
using LinearAlgebra
using JuMP
using GLPK
using Random

include("{julia_script_path}")

img = "{input_image}"
img_out = "{output_cs}"
tax = {tax}

img_reconstructed, error = compress_image_cs_l1(img, tax, img_out)
println("CS L1 concluido! MSE: ", error)
'''
    
    with open("temp_cs.jl", "w") as f:
        f.write(julia_code)
    
    try:
        result = subprocess.run(
            ["julia", "temp_cs.jl"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  ERRO: {result.stderr}")
            return False
        
        print(f"  OK: {output_cs}")
        return True
    except Exception as e:
        print(f"  ERRO: {e}")
        return False
    finally:
        if os.path.exists("temp_cs.jl"):
            os.remove("temp_cs.jl")


def run_ai_refiner(cs_image, output_final, model_path="cs_refiner_model.pth"):
    """
    Aplica refinamento IA na imagem CS
    """
    print(f"2. Aplicando refinamento IA...")
    
    if not os.path.exists(model_path):
        print(f"  ERRO: Modelo {model_path} nao encontrado!")
        print("  Treine primeiro: python cs_train_dataset.py")
        return False
    
    try:
        from cs_apply_model import load_model, refine_image
        
        model, device = load_model(model_path)
        refine_image(model, device, cs_image, output_final)
        
        print(f"  OK: {output_final}")
        return True
        
    except Exception as e:
        print(f"  ERRO: {e}")
        return False


def pipeline_completo(input_image, output_final, tax=0.5, model_path="cs_refiner_model.pth", keep_cs=False):
    """
    Pipeline completo: Original -> CS L1 -> IA Refiner
    """
    print("=" * 50)
    print("PIPELINE COMPLETO: Original -> CS L1 -> IA")
    print("=" * 50)
    
    if not os.path.exists(input_image):
        print(f"ERRO: Imagem {input_image} nao encontrada!")
        return False
    
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    cs_image = f"{base_name}_cs_l1.jpg"
    
    if not run_cs_l1(input_image, cs_image, tax):
        return False
    
    if not run_ai_refiner(cs_image, output_final, model_path):
        return False
    
    if not keep_cs:
        if os.path.exists(cs_image):
            os.remove(cs_image)
            print(f"  Imagem CS intermediaria removida")
    
    print("=" * 50)
    print("PIPELINE CONCLUIDO!")
    print(f"Original: {input_image}")
    print(f"Final:   {output_final}")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python cs_pipeline.py <imagem_original.jpg> <saida_final.jpg> [taxa] [modelo.pth]")
        print("\nExemplo:")
        print("  python cs_pipeline.py img.jpg img_final.jpg")
        print("  python cs_pipeline.py img.jpg img_final.jpg 0.5 cs_refiner_model.pth")
        print("\nParametros:")
        print("  taxa: Taxa de amostragem CS (0.0-1.0, padrao: 0.5)")
        print("  modelo: Caminho do modelo treinado (padrao: cs_refiner_model.pth)")
        exit(1)
    
    input_image = sys.argv[1]
    output_final = sys.argv[2]
    tax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    model_path = sys.argv[4] if len(sys.argv) > 4 else "cs_refiner_model.pth"
    
    pipeline_completo(input_image, output_final, tax, model_path, keep_cs=True)


