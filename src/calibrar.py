"""
=============================================================================
CALIBRAÇÃO — dois cliques sobre as marcações da zona de medição
=============================================================================

Abre um frame do vídeo numa janela AJUSTADA À TELA, recebe dois cliques sobre as
marcações e grava calibracao/<video>.json, usado por src/analisar_cruzamentos.py.

Como a janela mostra o vídeo reduzido, cada clique é convertido de volta para a
coordenada do vídeo original (divide pela escala de exibição). A resolução do
clique fica registrada no JSON. Para clicar com precisão, a lupa no canto superior
esquerdo mostra, em resolução ORIGINAL, a região em volta do cursor.

Comandos:
  clique esquerdo = marcar ponto (máximo 2)
  R = refazer os pontos
  C = confirmar e salvar
  Q = sair sem salvar

Uso (a partir da pasta tcc-velocidade):

    python src/calibrar.py --video videos/J_Correndo.mp4 --distancia 20
=============================================================================
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

from exibicao import calcular_escala_exibicao, reduzir_para_exibicao


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

VIDEO = "videos/J_Correndo.mp4"

# Distância REAL entre as duas marcações, em metros.
DISTANCIA_REAL_METROS = 20.0

PASTA_CALIBRACAO = "calibracao"

# Lupa: lado da região original mostrada e fator de ampliação.
LADO_LUPA_PX = 48
FATOR_LUPA = 6

NOME_JANELA = "Calibracao - clique nas duas marcacoes"


# =============================================================================
# ESTADO DO MOUSE
# =============================================================================

estado = {
    "pontos": [],       # em coordenadas do vídeo original
    "cursor": None,     # em coordenadas do vídeo original
    "escala": 1.0,
}


def ao_mouse(evento, x, y, flags, param):
    x_original = x / estado["escala"]
    y_original = y / estado["escala"]

    estado["cursor"] = (x_original, y_original)

    if evento == cv2.EVENT_LBUTTONUP and len(estado["pontos"]) < 2:
        estado["pontos"].append((round(x_original, 1), round(y_original, 1)))
        print(f"Marcação {len(estado['pontos'])}: X={x_original:.1f}, Y={y_original:.1f} "
              f"(px do vídeo original)")


# =============================================================================
# DESENHO (só exibição)
# =============================================================================

def desenhar_lupa(tela, frame, cursor):
    """Recorte do frame ORIGINAL em volta do cursor, ampliado, no canto da tela."""
    if cursor is None:
        return

    altura, largura = frame.shape[:2]
    meio = LADO_LUPA_PX // 2
    cx = int(round(cursor[0]))
    cy = int(round(cursor[1]))

    x0 = min(max(cx - meio, 0), largura - LADO_LUPA_PX)
    y0 = min(max(cy - meio, 0), altura - LADO_LUPA_PX)

    recorte = frame[y0:y0 + LADO_LUPA_PX, x0:x0 + LADO_LUPA_PX]
    lupa = cv2.resize(recorte, None, fx=FATOR_LUPA, fy=FATOR_LUPA,
                      interpolation=cv2.INTER_NEAREST)

    # Mira na posição exata do cursor dentro do recorte.
    mx = int((cursor[0] - x0) * FATOR_LUPA)
    my = int((cursor[1] - y0) * FATOR_LUPA)
    cv2.line(lupa, (mx, 0), (mx, lupa.shape[0]), (0, 0, 255), 1)
    cv2.line(lupa, (0, my), (lupa.shape[1], my), (0, 0, 255), 1)
    cv2.rectangle(lupa, (0, 0), (lupa.shape[1] - 1, lupa.shape[0] - 1), (255, 255, 255), 2)
    cv2.putText(lupa, f"x={cursor[0]:.1f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2)

    lado = min(lupa.shape[0], tela.shape[0] - 10, tela.shape[1] - 10)
    tela[10:10 + lado, 10:10 + lado] = lupa[:lado, :lado]


def montar_tela(frame, escala):
    tela = reduzir_para_exibicao(frame, escala).copy()

    for x, y in estado["pontos"]:
        xe, ye = int(round(x * escala)), int(round(y * escala))
        cv2.line(tela, (xe, 0), (xe, tela.shape[0]), (0, 0, 255), 1)
        cv2.circle(tela, (xe, ye), 5, (0, 255, 0), -1)

    desenhar_lupa(tela, frame, estado["cursor"])

    texto = (f"{len(estado['pontos'])}/2 pontos | C = confirmar | R = refazer | Q = sair")
    cv2.putText(tela, texto, (10, tela.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    return tela


# =============================================================================
# PRINCIPAL
# =============================================================================

def calibrar(video, distancia, numero_frame):
    caminho_video = Path(video)
    if not caminho_video.is_absolute():
        caminho_video = PASTA_PROJETO / caminho_video

    cap = cv2.VideoCapture(str(caminho_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, numero_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise SystemExit(f"Não foi possível ler o frame {numero_frame} de {caminho_video}")

    altura, largura = frame.shape[:2]
    estado["escala"] = calcular_escala_exibicao(largura, altura)

    print(f"Vídeo {caminho_video.name}: {largura}x{altura}, exibido a "
          f"{estado['escala'] * 100:.0f}% (1 px da tela = {1 / estado['escala']:.2f} px do vídeo)")
    print(f"Clique nas duas marcações separadas por {distancia} m. C confirma, R refaz, Q sai.")

    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(NOME_JANELA, ao_mouse)

    while True:
        cv2.imshow(NOME_JANELA, montar_tela(frame, estado["escala"]))
        tecla = cv2.waitKey(20) & 0xFF

        if tecla == ord("r"):
            estado["pontos"] = []
            print("Pontos apagados.")

        elif tecla == ord("c") and len(estado["pontos"]) == 2:
            break

        elif tecla == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Calibração cancelada; nada foi salvo.")

    cv2.destroyAllWindows()

    (x1, y1), (x2, y2) = estado["pontos"]
    distancia_px = abs(x2 - x1)

    if distancia_px == 0:
        raise SystemExit("Os dois pontos têm o mesmo X.")

    calibracao = {
        "video": caminho_video.name,
        "frame_calibracao": numero_frame,
        "pontos_px": [[x1, y1], [x2, y2]],
        "x_linhas_px": sorted([x1, x2]),
        "distancia_real_m": distancia,
        "distancia_px": distancia_px,
        "metros_por_pixel": distancia / distancia_px,
        "resolucao_clique_px": round(1 / estado["escala"], 3),
        "origem": "cliques em src/calibrar.py",
        "data": datetime.now().isoformat(timespec="seconds"),
    }

    caminho_saida = PASTA_PROJETO / PASTA_CALIBRACAO / f"{caminho_video.stem}.json"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    if caminho_saida.exists():
        anterior = json.loads(caminho_saida.read_text(encoding="utf-8"))
        print(f"\nSubstituindo calibração anterior: x = {anterior.get('x_linhas_px')} "
              f"({anterior.get('origem')})")

    caminho_saida.write_text(json.dumps(calibracao, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nx = {calibracao['x_linhas_px']} | {distancia_px:.1f} px = {distancia} m | "
          f"{calibracao['metros_por_pixel'] * 100:.4f} cm/px")
    print(f"Salvo em {caminho_saida}")


def main():
    parser = argparse.ArgumentParser(description="Calibração por dois cliques.")
    parser.add_argument("--video", default=VIDEO)
    parser.add_argument("--distancia", type=float, default=DISTANCIA_REAL_METROS,
                        help="distância real entre as marcações (m)")
    parser.add_argument("--frame", type=int, default=0, help="frame usado na calibração")
    args = parser.parse_args()

    calibrar(args.video, args.distancia, args.frame)


if __name__ == "__main__":
    main()
