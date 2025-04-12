import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import pyttsx3
from deep_translator import GoogleTranslator

# === Carrega o modelo BLIP ===
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# === Inicializa o motor de voz e tenta selecionar voz em português ===
voz = pyttsx3.init()
voz.setProperty('rate', 150)

for v in voz.getProperty('voices'):
    if 'portuguese' in v.name.lower() or 'brazil' in v.name.lower():
        voz.setProperty('voice', v.id)
        break

def falar(texto):
    print("🔊 Falando:", texto)
    voz.say(texto)
    voz.runAndWait()

def tirar_foto(nome_arquivo='foto.jpg'):
    cap = cv2.VideoCapture(1)  # ou 1, se sua webcam for externa

    if not cap.isOpened():
        print("Erro ao abrir a câmera.")
        return None

    print("Pressione qualquer tecla para tirar a foto...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem")
            break

        cv2.imshow("Aperte qualquer tecla para capturar", frame)

        if cv2.waitKey(1) != -1:
            cv2.imwrite(nome_arquivo, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"📸 Foto salva como {nome_arquivo}")
    return nome_arquivo

def descrever_imagem(caminho_imagem):
    image = Image.open(caminho_imagem).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**inputs)

    descricao_em_ingles = processor.decode(out[0], skip_special_tokens=True)
    return descricao_em_ingles

def traduzir_para_portugues(texto):
    try:
        return GoogleTranslator(source='en', target='pt').translate(texto)
    except Exception as e:
        print("Erro na tradução:", e)
        return texto  # Retorna original se falhar

# === Loop principal ===
while True:
    comando = input("\nDigite 'foto' para tirar uma foto ou 'sair' para encerrar: ").strip().lower()

    if comando == 'foto':
        caminho = tirar_foto()
        if caminho:
            descricao_ingles = descrever_imagem(caminho)
            print("\n📷 Descrição gerada (inglês):", descricao_ingles)

            descricao_pt = traduzir_para_portugues(descricao_ingles)
            print("📜 Descrição em português:", descricao_pt)

            falar(descricao_pt)
    elif comando == 'sair':
        print("Encerrando...")
        break
    else:
        print("Comando não reconhecido. Use 'foto' ou 'sair'.")
