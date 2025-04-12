import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Carrega o modelo BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def tirar_foto(nome_arquivo='foto.jpg'):
    # Abre a câmera (0 geralmente é a câmera USB padrão)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return None

    print("Pressione qualquer tecla para tirar a foto...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem")
            break

        # Mostra a imagem em tempo real
        cv2.imshow("Aperte qualquer tecla para capturar", frame)

        # Espera tecla
        if cv2.waitKey(1) != -1:
            cv2.imwrite(nome_arquivo, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Foto salva como {nome_arquivo}")
    return nome_arquivo

def descrever_imagem(caminho_imagem):
    image = Image.open(caminho_imagem).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)

    descricao = processor.decode(out[0], skip_special_tokens=True)
    return descricao

# === Loop principal para tirar e descrever ===
while True:
    comando = input("\nDigite 'foto' para tirar uma foto ou 'sair' para encerrar: ").strip().lower()

    if comando == 'foto':
        caminho = tirar_foto()
        if caminho:
            descricao = descrever_imagem(caminho)
            print("\n📷 Descrição gerada:", descricao)
    elif comando == 'sair':
        print("Encerrando...")
        break
    else:
        print("Comando não reconhecido. Use 'foto' ou 'sair'.")
