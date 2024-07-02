import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageTk
import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import Label
import time

# Cargar el modelo ResNet preentrenado con weights
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model.eval()

# Quitar la capa de clasificación para obtener solo las características
model = nn.Sequential(*list(model.children())[:-1])

# Transformaciones para preprocesar las imágenes
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
])

# Función para extraer características de una imagen
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)
    
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

# Función para encontrar la imagen más similar en la carpeta
def find_most_similar(captured_features, folder_path):
    min_distance = float('inf')
    most_similar_image = None
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
            image_path = os.path.join(folder_path, filename)
            image_features = extract_features(image_path)
            distance = np.linalg.norm(captured_features - image_features)
            
            if distance < min_distance:
                min_distance = distance
                most_similar_image = image_path
                
    return most_similar_image

# Función para capturar imagen desde la webcam y realizar el reconocimiento de objetos
def capture_image_from_webcam(image_path: str):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    def show_frame():
        ret, frame = cap.read()
        if ret:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            label.after(10, show_frame)
    
    root = tk.Tk()
    label = Label(root)
    label.pack()
    show_frame()
    
    root.after(5000, lambda: root.destroy())  # Cerrar la ventana después de 5 segundos
    root.mainloop()
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
        print(f"Image saved to {image_path}")
    else:
        print("Error: Could not capture image")

    cap.release()

# Mostrar la imagen recomendada
def show_image(image_path):
    if image_path is None:
        print("No se encontró ninguna imagen similar.")
        return
    
    img = Image.open(image_path)
    img.show()

# Capturar imagen desde la webcam
captured_image_path = 'captured_image.jpg'
capture_image_from_webcam(captured_image_path)

if os.path.exists(captured_image_path):
    # Extraer características de la imagen capturada
    captured_features = extract_features(captured_image_path)

    # Ruta a la carpeta con las imágenes de productos
    folder_path = 'productos'

    # Encontrar la imagen más similar en la carpeta
    most_similar_image = find_most_similar(captured_features, folder_path)
    if most_similar_image:
        print(f'Imagen más similar: {most_similar_image}')
        # Mostrar la imagen recomendada
        show_image(most_similar_image)
    else:
        print("No se encontró ninguna imagen similar.")
else:
    print("No se capturó ninguna imagen.")
