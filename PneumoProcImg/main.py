import cv2
import os
import zipfile

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


                    
    