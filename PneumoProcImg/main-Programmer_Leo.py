import cv2
import os
import zipfile
import torch 

zip_file = "PneumoProcImg/pneumonia.zip"
ext = "PneumoProcImg/chest_xray"

if not os.path.exists(ext):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(ext)
else:
    print("Já extraído!")        

def process_img(input_dir, output_dir, target_size = (224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for folder in os.listdir(input_dir):
        class_path = os.path.join(input_dir, folder)
        output_class_path = os.path.join(output_dir, folder)
        os.makedirs(output_class_path, exist_ok=True)
        
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_res = cv2.resize(img, target_size)
                    cv2.imwrite(os.path.join(output_class_path, filename), img_res)
                    
                    

input_dirs = ["PneumoProcImg/chest_xray1/train", "PneumoProcImg/chest_xray1/test"]
output_dirs = ["processed_data/train", "processed_data/test"]



for input_dir, output_dir in zip(input_dirs, output_dirs):
    process_img(input_dir, output_dir)
print("Imagens processadas e salvas")

                                            


                    
    