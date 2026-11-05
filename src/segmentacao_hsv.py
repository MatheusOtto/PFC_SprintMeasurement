"""
=============================================================================
SEGMENTAÇÃO POR COR (HSV) NO ENSAIO A
=============================================================================

Primeira abordagem de detecção do corredor: o ponto medido é o centro da maior região laranja
(o colete) em cada quadro. O ruído, os descartes e a janela do trecho são calculados pelas mesmas
funções usadas com a estimação de pose, em analisar_cruzamentos.medir(), sem filtro de Kalman.
Sobre a posição horizontal do ponto, o programa imprime:

  - deslocamento entre quadros consecutivos, no sentido da corrida, e sua mediana;
  - saltos: deslocamentos várias vezes maiores que a mediana e quadros em que o ponto recua;
  - segundas diferenças d, as que passam do limite de 5 desvios padrão e o ruído do ponto.

As posições detectadas ficam em resultados/hsv/Trotando_2_hsv.csv. Com --video, a detecção é
refeita a partir do vídeo e o CSV é regravado.

Uso (na raiz do repositório):

    python src/segmentacao_hsv.py
    python src/segmentacao_hsv.py --video caminho/do/video.mp4
=============================================================================
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PASTA_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_PROJETO / "src"))

import analisar_cruzamentos as ac  # noqa: E402

ENSAIO = "Trotando_2"
ARQUIVO_POSICOES = PASTA_PROJETO / "resultados" / "hsv" / f"{ENSAIO}_hsv.csv"

# Faixa de cor do colete, no espaço HSV do OpenCV, e área mínima da região, em pixels².
HSV_MIN = np.array([0, 168, 147])
HSV_MAX = np.array([176, 255, 255])
AREA_MINIMA = 100

SALTO_MINIMO_PX = 40       # deslocamento considerado salto


def detectar(frame):
    """Centro (x, y) da maior região dentro da faixa de cor; None se não houver região válida."""
    mascara = cv2.dilate(cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), HSV_MIN, HSV_MAX), None, iterations=2)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        momentos = cv2.moments(maior)
        if cv2.contourArea(maior) > AREA_MINIMA and momentos["m00"] > 0:
            return momentos["m10"] / momentos["m00"], momentos["m01"] / momentos["m00"]
    return None


def gravar_posicoes(caminho_video, quadros):
    """Percorre o vídeo e grava a posição detectada em cada quadro."""
    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise SystemExit(f"Não abri {caminho_video}")

    ARQUIVO_POSICOES.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_POSICOES, "w", newline="", encoding="utf-8") as arquivo:
        saida = csv.writer(arquivo)
        saida.writerow(["frame", "cx", "cy"])
        for quadro in quadros:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"O vídeo acabou no quadro {quadro}.")
            centro = detectar(frame)
            saida.writerow([int(quadro), *(centro if centro is not None else ("", ""))])
    cap.release()


def medir_hsv():
    """Mede o ensaio com as posições da cor, usando os carimbos de tempo do CSV de keypoints."""
    calibracao = json.loads((PASTA_PROJETO / "calibracao" / f"{ENSAIO}.json").read_text(encoding="utf-8"))
    dados = dict(ac.carregar_keypoints(ac.csv_maior_imgsz(ENSAIO)))

    with open(ARQUIVO_POSICOES, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    dados["cx"] = np.array([float(l["cx"]) if l["cx"] else np.nan for l in linhas])
    dados["n_pessoas"] = None

    return dados, ac.medir(dados, calibracao, ac.referencia_cronometro(ENSAIO), com_kalman=False)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Segmentação por cor no Ensaio A.")
    parser.add_argument("--video", default=None, help="refaz a detecção a partir deste vídeo")
    args = parser.parse_args()

    if args.video is not None:
        gravar_posicoes(args.video, ac.carregar_keypoints(ac.csv_maior_imgsz(ENSAIO))["frame"])

    dados, m = medir_hsv()

    # Quadros do trecho usados no ruído: da linha de entrada (antes) à de saída (depois).
    q = dados["frame"]
    janela = (q >= m["entrada"]["frame_antes"]) & (q <= m["saida"]["frame_depois"])
    q, x = q[janela], dados["cx"][janela]

    # Passo positivo é avançar; negativo é recuar.
    passo = m["direcao"] * np.diff(x)
    mediana = np.median(passo)
    descartados = ~np.isin(m["d"], m["d_usados"])

    print(f"{ENSAIO}: segmentação por cor, trecho dos quadros {q[0]} a {q[-1]}")
    print(f"deslocamento entre quadros: mediana {mediana:.1f} px; mín {passo.min():.1f}, "
          f"máx {passo.max():.1f}")
    saltos = np.flatnonzero(np.abs(passo) > SALTO_MINIMO_PX)
    print(f"saltos acima de {SALTO_MINIMO_PX} px: "
          + ", ".join(f"{q[i]}->{q[i + 1]} ({passo[i]:+.1f} px, {abs(passo[i]) / mediana:.1f}× a mediana)"
                      for i in saltos))
    recuos = np.flatnonzero(passo < 0)
    print("pares em que o ponto recua: "
          + ", ".join(f"{q[i]}->{q[i + 1]} ({passo[i]:+.1f} px)" for i in recuos))
    print(f"segundas diferenças d: {len(m['d'])}; acima do limite de {ac.CORTE_GROSSEIRO_DESVIOS:g} desvios: "
          f"{int(descartados.sum())} de {len(m['d'])}")
    print(f"ruído do ponto: σx = {m['sigma_x_px']:.2f} px")
    print(f"velocidade média no trecho: {m['v_media']:.4f} m/s")


if __name__ == "__main__":
    main()
