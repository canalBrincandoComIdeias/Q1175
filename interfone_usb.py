#!/usr/bin/env python3
# Casa Inteligente do Zero - TM2
# Ouve um microfone USB (qualquer marca), roda o modelo e avisa o Home Assistant via MQTT.
# O programa se adapta ao microfone: descobre a taxa que ele aceita e converte sozinho.

import os, sys, json
import numpy as np
import pyaudio
import paho.mqtt.client as mqtt
from edge_impulse_linux.runner import ImpulseRunner

# ================== CONFIGURAÇÃO ==================
MODELO_ARQUIVO = "interfone.eim"   # o modelo baixado do Edge Impulse
CLASSE         = "interfone"       # a classe que queremos detectar

MQTT_HOST    = "192.168.50.10"     # IP do Home Assistant
MQTT_PORTA   = 1883
MQTT_USUARIO = "mqtt"
MQTT_SENHA   = "SENHA_AQUI"

LIMIAR_LIGA      = 0.80   # certeza mínima para contar como detectado
LIMIAR_DESLIGA   = 0.50   # abaixo disso, conta como silêncio
CONFIRMA_LIGA    = 3      # leituras seguidas para LIGAR
CONFIRMA_DESLIGA = 6      # leituras seguidas para DESLIGAR
# ==================================================

PASTA  = os.path.dirname(os.path.realpath(__file__))
MODELO = os.path.join(PASTA, MODELO_ARQUIVO)

TOPICO_CONFIG = "homeassistant/binary_sensor/interfone/config"
TOPICO_ESTADO = "casa-que-ouve-unoq/interfone/estado"
TOPICO_ATRIB  = "casa-que-ouve-unoq/interfone/atributos"
TOPICO_DISP   = "casa-que-ouve-unoq/disponivel"

config_sensor = {
    "name": "Interfone",
    "unique_id": "casa_que_ouve_interfone",
    "state_topic": TOPICO_ESTADO,
    "json_attributes_topic": TOPICO_ATRIB,
    "availability_topic": TOPICO_DISP,
    "device_class": "sound",
    "icon": "mdi:doorbell",
    "payload_on": "ON",
    "payload_off": "OFF",
    "device": {
        "identifiers": ["casa_que_ouve_uno_q"],
        "name": "Casa que Ouve (UNO Q)",
        "model": "Edge Impulse + Arduino UNO Q",
    },
}

# ---------- Microfone ----------
def achar_microfone_usb(pa):
    """Procura o primeiro microfone com 'USB' no nome."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0 and "usb" in info.get("name", "").lower():
            return i, info.get("name")
    return None, None

def escolher_formato(pa, indice):
    """Descobre uma taxa e um número de canais que o microfone aceita."""
    for canais in (1, 2):
        for taxa in (16000, 48000, 44100, 32000, 22050):
            try:
                if pa.is_format_supported(taxa, input_device=indice,
                                          input_channels=canais,
                                          input_format=pyaudio.paInt16):
                    return taxa, canais
            except ValueError:
                pass
    return None, None

def converter_taxa(amostras, taxa_origem, taxa_destino):
    """Converte o áudio da taxa do microfone para a taxa do modelo."""
    if taxa_origem == taxa_destino:
        return amostras
    fator = taxa_origem / taxa_destino
    k = int(np.ceil(fator))
    if k > 1:  # suaviza antes de reduzir, para não distorcer
        amostras = np.convolve(amostras, np.ones(k) / k, mode="same")
    n_saida = int(round(len(amostras) / fator))
    x_orig = np.arange(len(amostras))
    x_novo = np.linspace(0, len(amostras) - 1, n_saida)
    return np.interp(x_novo, x_orig, amostras)

# ---------- MQTT ----------
def publicar_estado(cliente, estado, certeza):
    cliente.publish(TOPICO_ESTADO, estado, retain=True)
    cliente.publish(TOPICO_ATRIB, json.dumps({"certeza": round(certeza, 3)}), retain=True)
    print(f">>> Sensor agora: {estado} (certeza {certeza:.2f})", flush=True)

cliente = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="casa-que-ouve-unoq")
cliente.username_pw_set(MQTT_USUARIO, MQTT_SENHA)
cliente.will_set(TOPICO_DISP, "offline", retain=True)
cliente.connect(MQTT_HOST, MQTT_PORTA)
cliente.loop_start()
cliente.publish(TOPICO_CONFIG, json.dumps(config_sensor), retain=True)
cliente.publish(TOPICO_DISP, "online", retain=True)

# ---------- Modelo ----------
runner = ImpulseRunner(MODELO)
info = runner.init()
params = info["model_parameters"]
TAXA_MODELO = int(params["frequency"])            # ex.: 16000
JANELA      = int(params["input_features_count"]) # ex.: 16000 amostras = 1 segundo
print(f"Modelo: taxa {TAXA_MODELO} Hz, janela {JANELA} amostras, classes {params['labels']}", flush=True)

# ---------- Abre o microfone ----------
pa = pyaudio.PyAudio()
indice, nome = achar_microfone_usb(pa)
if indice is None:
    print("Nenhum microfone USB encontrado. Ligue um microfone USB e tente de novo.")
    sys.exit(1)
taxa, canais = escolher_formato(pa, indice)
if taxa is None:
    print(f"O microfone '{nome}' não aceitou nenhum formato conhecido.")
    sys.exit(1)
print(f"Microfone: {nome} | taxa {taxa} Hz | {canais} canal(is)", flush=True)

bloco = taxa // 4   # lê 1/4 de segundo por vez
stream = pa.open(format=pyaudio.paInt16, channels=canais, rate=taxa,
                 input=True, input_device_index=indice, frames_per_buffer=bloco)

estado = "OFF"
contador = 0
publicar_estado(cliente, estado, 0.0)
buffer = np.zeros(0, dtype=np.float32)

print("Ouvindo... (Ctrl+C para sair)", flush=True)
try:
    while True:
        dados = stream.read(bloco, exception_on_overflow=False)
        a = np.frombuffer(dados, dtype=np.int16).astype(np.float32)
        if canais == 2:
            a = a.reshape(-1, 2).mean(axis=1)          # junta os dois canais
        a = converter_taxa(a, taxa, TAXA_MODELO)
        buffer = np.concatenate([buffer, a])[-JANELA:] # guarda só o último segundo
        if len(buffer) < JANELA:
            continue

        amostras = np.clip(np.round(buffer), -32768, 32767).astype(np.int16).tolist()
        resultado = runner.classify(amostras)
        p = resultado["result"]["classification"][CLASSE]

        if estado == "OFF":
            contador = contador + 1 if p >= LIMIAR_LIGA else 0
            if contador >= CONFIRMA_LIGA:
                estado, contador = "ON", 0
                publicar_estado(cliente, estado, p)
        else:
            contador = contador + 1 if p < LIMIAR_DESLIGA else 0
            if contador >= CONFIRMA_DESLIGA:
                estado, contador = "OFF", 0
                publicar_estado(cliente, estado, p)

        print(f"{CLASSE}={p:.2f}  estado={estado}  contador={contador}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    cliente.publish(TOPICO_DISP, "offline", retain=True)
    cliente.loop_stop()
    cliente.disconnect()
    stream.stop_stream(); stream.close(); pa.terminate()
    runner.stop()
    print("Encerrado.", flush=True)