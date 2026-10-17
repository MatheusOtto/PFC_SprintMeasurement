"""
=============================================================================
ETAPA 1 DO PIPELINE — EXTRAÇÃO DOS KEYPOINTS (vídeo → CSV bruto por frame)
=============================================================================

Roda o YOLO pose em todos os frames e grava, para cada um:

    frame, t_ms (carimbo real do container), t_nominal_s (frame / FPS),
    número de pessoas, confiança e bbox da pessoa escolhida,
    x, y e confiança dos 17 keypoints COCO.

Não calcula velocidade nem faz calibração. A ideia é rodar a inferência UMA vez
e permitir que qualquer critério de análise (limiar de confiança, interpolação,
gap máximo) seja reavaliado sobre os mesmos dados, sem rodar o YOLO de novo.

Junto do CSV é gravado um JSON com tudo o que é preciso para reproduzir a
extração: versões das bibliotecas, md5 dos pesos, imgsz, limiar de detecção.
Isso existe porque a MESMA combinação vídeo + pesos deu detecções diferentes em
ultralytics 8.4.133 e 8.4.174 (ver diagnostico/J_Correndo.md).

Uso (a partir da pasta tcc-velocidade):

    python src/extrair_keypoints.py                       # padrões abaixo
    python src/extrair_keypoints.py --imgsz 1920
    python src/extrair_keypoints.py --video videos/X.mp4 --mostrar

Sem --mostrar roda sem janela (mais rápido e pode rodar em segundo plano).
Com --mostrar, Q interrompe; o CSV parcial fica salvo e marcado no JSON.

Diferenças em relação ao legado src/Testar_Keypoints_YOLO.py:
  - grava t_ms real (CAP_PROP_POS_MSEC) além do tempo nominal;
  - imgsz e conf de detecção explícitos e registrados;
  - grava n_pessoas (quantas pessoas o YOLO viu no frame);
  - caminhos relativos à pasta do projeto, não à pasta de execução.
A seleção da pessoa (maior bbox) e o formato das colunas de keypoints são os
mesmos do legado.
=============================================================================
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from exibicao import calcular_escala_exibicao, reduzir_para_exibicao


# =============================================================================
# CONFIGURAÇÕES (padrões; a linha de comando pode sobrescrever)
# =============================================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

VIDEO = "videos/J_Correndo.mp4"

MODELO_POSE = "yolo26n-pose.pt"

# Lado da imagem de entrada do modelo.
# "nativo" = maior lado do vídeo (arredondado para múltiplo de 32): o frame entra
# no modelo sem redução e sem ampliação. Escolhido pelo experimento
# experimentos/2026-10-07-imgsz-J_Correndo.md: com 640 (padrão do ultralytics) um
# vídeo 1920×1080 é reduzido 3× e o corredor a 21 m deixa de ser detectado.
# Também aceita um número (ex.: --imgsz 640 reproduz o legado).
IMGSZ = "nativo"

# Limiar de confiança da PESSOA (detecção). 0,25 é o padrão do ultralytics;
# fica explícito para não mudar silenciosamente se o padrão da biblioteca mudar.
CONF_DETECCAO = 0.25

PASTA_SAIDA = "resultados/keypoints"

NOMES_KEYPOINTS = [
    "nariz",              # 0
    "olho_esquerdo",      # 1
    "olho_direito",       # 2
    "orelha_esquerda",    # 3
    "orelha_direita",     # 4
    "ombro_esquerdo",     # 5
    "ombro_direito",      # 6
    "cotovelo_esquerdo",  # 7
    "cotovelo_direito",   # 8
    "pulso_esquerdo",     # 9
    "pulso_direito",      # 10
    "quadril_esquerdo",   # 11
    "quadril_direito",    # 12
    "joelho_esquerdo",    # 13
    "joelho_direito",     # 14
    "tornozelo_esquerdo", # 15
    "tornozelo_direito",  # 16
]

# Só para a janela de acompanhamento (--mostrar).
LIMIAR_EXIBICAO = 0.35


# =============================================================================
# AUXILIARES
# =============================================================================

def resolver(caminho):
    """Caminho relativo é relativo à pasta do projeto, não à pasta de execução."""
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho


def md5_arquivo(caminho):
    h = hashlib.md5()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def criar_cabecalho():
    cabecalho = [
        "frame",
        "t_ms",
        "t_nominal_s",
        "n_pessoas",
        "pessoa_detectada",
        "confianca_pessoa",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    for indice, nome in enumerate(NOMES_KEYPOINTS):
        cabecalho.extend([
            f"kp{indice}_{nome}_x",
            f"kp{indice}_{nome}_y",
            f"kp{indice}_{nome}_conf",
        ])

    return cabecalho


def selecionar_pessoa_principal(resultado):
    """
    Maior bounding box, como no legado. Adequado para um corredor principal;
    não é tracking. n_pessoas fica gravado para que a análise possa desconfiar
    de frames com mais de uma pessoa.
    """
    if resultado.boxes is None or resultado.keypoints is None:
        return 0, None

    n_pessoas = len(resultado.boxes)

    if n_pessoas == 0:
        return 0, None

    caixas = resultado.boxes.xyxy.cpu().numpy()
    keypoints = resultado.keypoints.data.cpu().numpy()
    confiancas = resultado.boxes.conf.cpu().numpy()

    areas = (caixas[:, 2] - caixas[:, 0]) * (caixas[:, 3] - caixas[:, 1])
    indice = int(np.argmax(areas))

    return n_pessoas, (caixas[indice], keypoints[indice], float(confiancas[indice]))


def criar_linha(frame_num, t_ms, t_nominal_s, n_pessoas, pessoa):
    linha = [frame_num, t_ms, t_nominal_s, n_pessoas]

    if pessoa is None:
        linha.extend([0, "", "", "", "", ""])
        linha.extend([""] * (3 * len(NOMES_KEYPOINTS)))
        return linha

    caixa, pontos, confianca = pessoa
    linha.extend([1, confianca, *[float(v) for v in caixa]])

    for ponto in pontos:
        linha.extend([float(ponto[0]), float(ponto[1]), float(ponto[2])])

    return linha


def desenhar(frame, pessoa, texto):
    """Desenha numa cópia. Só para a janela de acompanhamento."""
    tela = frame.copy()

    if pessoa is not None:
        caixa, pontos, confianca = pessoa
        x1, y1, x2, y2 = caixa.astype(int)
        cv2.rectangle(tela, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for indice, ponto in enumerate(pontos):
            x, y, conf = float(ponto[0]), float(ponto[1]), float(ponto[2])
            if x <= 0 or y <= 0:
                continue
            cor = (0, 255, 255) if conf >= LIMIAR_EXIBICAO else (0, 0, 255)
            raio = 6 if indice in (5, 6) else 3
            cv2.circle(tela, (int(round(x)), int(round(y))), raio, cor, -1)

        cv2.putText(tela, f"pessoa {confianca:.2f}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(tela, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(tela, "Q = interromper", (30, tela.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return tela


# =============================================================================
# EXTRAÇÃO
# =============================================================================

def extrair(video, modelo_pose, imgsz, conf_deteccao, saida, mostrar, limite_frames):
    caminho_video = resolver(video)
    caminho_modelo = resolver(modelo_pose)

    if not caminho_video.exists():
        raise SystemExit(f"Vídeo não encontrado: {caminho_video}")

    if not caminho_modelo.exists():
        raise SystemExit(f"Pesos não encontrados: {caminho_modelo}")

    cap = cv2.VideoCapture(str(caminho_video))

    if not cap.isOpened():
        raise SystemExit(f"Erro ao abrir o vídeo: {caminho_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_container = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise SystemExit("FPS inválido no container.")

    if str(imgsz) == "nativo":
        imgsz = 32 * math.ceil(max(largura, altura) / 32)
    else:
        imgsz = int(imgsz)

    if saida is None:
        saida = Path(PASTA_SAIDA) / f"{caminho_video.stem}_imgsz{imgsz}.csv"

    caminho_csv = resolver(saida)
    caminho_json = caminho_csv.with_suffix(".json")
    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    modelo = YOLO(str(caminho_modelo))

    print(f"Vídeo:   {caminho_video.name}  {largura}x{altura}  {fps:.4f} fps  "
          f"{n_frames_container} frames")
    print(f"Modelo:  {caminho_modelo.name}  imgsz={imgsz}  conf={conf_deteccao}")
    print(f"Saída:   {caminho_csv}")

    nome_janela = f"Extracao de keypoints - imgsz {imgsz}"
    escala_tela = calcular_escala_exibicao(largura, altura) if mostrar else 1.0

    frame_num = 0
    frames_com_pessoa = 0
    interrompido = False
    inicio = time.perf_counter()

    with open(caminho_csv, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(criar_cabecalho())

        while True:
            if limite_frames is not None and frame_num >= limite_frames:
                break

            ok, frame = cap.read()

            if not ok:
                break

            # Depois do read(), POS_MSEC é o carimbo do frame que acabou de ser
            # lido (verificado: o primeiro frame retorna 0,0 ms).
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_nominal_s = frame_num / fps

            resultado = modelo.predict(
                frame,
                imgsz=imgsz,
                conf=conf_deteccao,
                verbose=False,
            )[0]

            n_pessoas, pessoa = selecionar_pessoa_principal(resultado)

            if pessoa is not None:
                frames_com_pessoa += 1

            escritor.writerow(criar_linha(frame_num, t_ms, t_nominal_s, n_pessoas, pessoa))

            if mostrar:
                texto = f"frame {frame_num}  t={t_ms / 1000:.3f}s  pessoas={n_pessoas}"
                tela = reduzir_para_exibicao(desenhar(frame, pessoa, texto), escala_tela)
                cv2.imshow(nome_janela, tela)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    interrompido = True
                    frame_num += 1
                    break

            frame_num += 1

            if frame_num % 100 == 0:
                decorrido = time.perf_counter() - inicio
                print(f"  {frame_num}/{n_frames_container} frames  "
                      f"({decorrido / frame_num * 1000:.0f} ms/frame, "
                      f"{frames_com_pessoa} com pessoa)", flush=True)

    duracao = time.perf_counter() - inicio
    cap.release()

    if mostrar:
        cv2.destroyAllWindows()

    metadados = {
        "video": caminho_video.name,
        "largura_px": largura,
        "altura_px": altura,
        "fps_nominal": fps,
        "n_frames_container": n_frames_container,
        "n_frames_processados": frame_num,
        "interrompido": interrompido,
        "frames_com_pessoa": frames_com_pessoa,
        "modelo": caminho_modelo.name,
        "modelo_md5": md5_arquivo(caminho_modelo),
        "imgsz": imgsz,
        "conf_deteccao": conf_deteccao,
        "selecao_pessoa": "maior bbox",
        "base_tempo": "t_ms = CAP_PROP_POS_MSEC após read(); t_nominal_s = frame / fps_nominal",
        "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
        "versoes": {
            "python": platform.python_version(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "data_extracao": datetime.now().isoformat(timespec="seconds"),
        "tempo_processamento_s": round(duracao, 1),
        "ms_por_frame": round(1000 * duracao / max(frame_num, 1), 1),
    }

    with open(caminho_json, "w", encoding="utf-8") as arquivo:
        json.dump(metadados, arquivo, ensure_ascii=False, indent=2)

    print(f"\nConcluído: {frame_num} frames, {frames_com_pessoa} com pessoa, "
          f"{duracao:.0f} s ({metadados['ms_por_frame']} ms/frame)"
          f"{'  [INTERROMPIDO]' if interrompido else ''}")
    print(f"CSV:  {caminho_csv}")
    print(f"JSON: {caminho_json}")


def main():
    parser = argparse.ArgumentParser(description="Extrai keypoints YOLO pose por frame.")
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--modelo", default=MODELO_POSE)
    parser.add_argument("--imgsz", default=IMGSZ,
                        help='"nativo" (maior lado do vídeo) ou um número, ex.: 640')
    parser.add_argument("--conf", type=float, default=CONF_DETECCAO)
    parser.add_argument("--saida", default=None,
                        help="CSV de saída (padrão: resultados/keypoints/<video>_imgsz<N>.csv)")
    parser.add_argument("--mostrar", action="store_true",
                        help="abre janela de acompanhamento ajustada à tela")
    parser.add_argument("--limite-frames", type=int, default=None,
                        help="processa só os N primeiros frames (para teste)")
    args = parser.parse_args()

    extrair(
        video=args.video,
        modelo_pose=args.modelo,
        imgsz=args.imgsz,
        conf_deteccao=args.conf,
        saida=args.saida,
        mostrar=args.mostrar,
        limite_frames=args.limite_frames,
    )


if __name__ == "__main__":
    main()
