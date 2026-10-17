"""
=============================================================================
VISUALIZAÇÃO DA MEDIÇÃO — vídeo com as medições sobrepostas
=============================================================================

Reproduz o vídeo mostrando, frame a frame:
  - as linhas da zona de medição e o centro dos ombros;
  - o tempo dentro do trecho e a velocidade instantânea;
  - um gráfico da velocidade instantânea ao longo do trecho: V bruta (pontos) e
    Kalman suavizado com faixa ±2σ;
  - no final: a velocidade média no trecho (distância / tempo) com o tempo do
    cronômetro e da visão computacional, sem e com filtro, e a V mínima e a máxima
    do Kalman suavizado no trecho.

NÃO roda o YOLO: lê o CSV de src/extrair_keypoints.py e calcula tudo com
analisar_cruzamentos.medir() (incluindo o Kalman). Os números da tela são os
mesmos do relatório; este arquivo não calcula nenhuma grandeza medida.
A abertura demora alguns segundos: é o ajuste do Kalman.

Comandos:
  ESPAÇO = pausar / continuar
  + / -  = mais rápido / mais lento
  R      = recomeçar
  S      = gravar a tela atual em PNG (monografia/figuras/tela_<video>_q<quadro>.png)
  Q      = sair

Uso (a partir da pasta tcc-velocidade):

    python src/visualizar_medicao.py                      # menu para escolher o vídeo
    python src/visualizar_medicao.py --video J_Variando
    python src/visualizar_medicao.py --video J_Variando --salvar-quadro   # só grava a tela final

A tela gravada tem 1280 px de largura, para o texto ficar legível na página.

O CSV usado é o de maior imgsz do vídeo; a calibração, calibracao/<video>.json; o
cronômetro, o último registrado em resultados/ensaios.csv (ou --t-cronometro).
=============================================================================
"""

import argparse
import csv
import json
import os
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analisar_cruzamentos import analisar, carregar_keypoints
from exibicao import calcular_escala_exibicao, reduzir_para_exibicao


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

PASTA_KEYPOINTS = "resultados/keypoints"
ENSAIOS = "resultados/ensaios.csv"

# Velocidade de reprodução (1.0 = tempo real). O trecho de 20 m dura ~3,4 s.
VELOCIDADE_REPRODUCAO = 0.5
VELOCIDADES_DISPONIVEIS = [0.1, 0.25, 0.5, 1.0, 2.0]

# Gráfico: faixa vertical (m/s) e altura relativa à janela.
V_MAX_GRAFICO = 12.0
ALTURA_GRAFICO = 0.28

# Lupa sobre o corredor: lado da região original e fator de ampliação.
LADO_LUPA_PX = 260
FATOR_LUPA = 1.2

NOME_JANELA = "Medicao - velocidade instantanea"

# Tela gravada para a monografia (--salvar-quadro ou tecla S): largura fixa, para
# que o texto da tela fique legível quando a figura ocupa a largura da página.
PASTA_FIGURAS = "monografia/figuras"
LARGURA_FIGURA = 1280

# Texto da tela: fonte TrueType (com acentos e ±). Sem ela, volta para a fonte do
# OpenCV, sem acentos. PIXELS_POR_ESCALA converte a escala do OpenCV em tamanho de
# fonte com altura parecida.
PASTA_FONTES = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FONTES = {False: ("segoeui.ttf", "arial.ttf"), True: ("segoeuib.ttf", "arialbd.ttf")}
PIXELS_POR_ESCALA = 31

# Cores (BGR)
VERMELHO = (0, 0, 255)
VERDE = (0, 255, 0)
AMARELO = (0, 255, 255)
CIANO = (255, 255, 0)
BRANCO = (255, 255, 255)
CINZA = (160, 160, 160)
AZUL = (255, 120, 0)


# =============================================================================
# AUXILIARES
# =============================================================================

def resolver(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho


@lru_cache(maxsize=None)
def fonte(tamanho, negrito):
    for nome in FONTES[negrito]:
        caminho = PASTA_FONTES / nome
        if caminho.exists():
            return ImageFont.truetype(str(caminho), tamanho)
    return None


def largura_texto(conteudo, escala=0.6, espessura=1):
    f = fonte(max(8, round(PIXELS_POR_ESCALA * escala)), espessura >= 2)
    if f is None:
        return cv2.getTextSize(conteudo, cv2.FONT_HERSHEY_SIMPLEX, escala, espessura)[0][0]
    return int(round(f.getlength(conteudo)))


def texto(img, conteudo, posicao, escala=0.6, cor=BRANCO, espessura=1):
    """
    Texto com contorno preto, legível sobre qualquer fundo. `posicao` é o início da
    linha de base, como no cv2.putText. Só o retângulo do texto passa pelo Pillow.
    """
    f = fonte(max(8, round(PIXELS_POR_ESCALA * escala)), espessura >= 2)

    if f is None:
        sem_acento = unicodedata.normalize("NFKD", conteudo.replace("±", "+-"))
        sem_acento = sem_acento.encode("ascii", "ignore").decode()
        cv2.putText(img, sem_acento, posicao, cv2.FONT_HERSHEY_SIMPLEX, escala,
                    (0, 0, 0), espessura + 3, cv2.LINE_AA)
        cv2.putText(img, sem_acento, posicao, cv2.FONT_HERSHEY_SIMPLEX, escala,
                    cor, espessura, cv2.LINE_AA)
        return

    contorno = 2
    x, y = posicao
    esq, topo, dir_, base = f.getbbox(conteudo, anchor="ls", stroke_width=contorno)
    x0, y0 = max(x + esq, 0), max(y + topo, 0)
    x1, y1 = min(x + dir_ + 1, img.shape[1]), min(y + base + 1, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    # A imagem é BGR e a cor também: o Pillow só copia os três canais na ordem dada.
    recorte = Image.fromarray(np.ascontiguousarray(img[y0:y1, x0:x1]))
    ImageDraw.Draw(recorte).text((x - x0, y - y0), conteudo, font=f, fill=tuple(cor),
                                 anchor="ls", stroke_width=contorno, stroke_fill=(0, 0, 0))
    img[y0:y1, x0:x1] = np.asarray(recorte)


def painel(img, x0, y0, x1, y1, opacidade=0.55):
    sobreposicao = img.copy()
    cv2.rectangle(sobreposicao, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(sobreposicao, opacidade, img, 1 - opacidade, 0, img)


def fmt(valor, casas=3):
    """Número com vírgula decimal, como no texto do TCC."""
    return f"{valor:.{casas}f}".replace(".", ",")


# =============================================================================
# DESENHO
# =============================================================================

def desenhar_no_original(frame, dados, medicao, i):
    """Elementos em coordenadas do vídeo original (antes de reduzir)."""
    altura, largura = frame.shape[:2]

    for x, rotulo in ((medicao["entrada"]["x_linha"], "ENTRADA"),
                      (medicao["saida"]["x_linha"], "SAÍDA")):
        xi = int(round(x))
        cv2.line(frame, (xi, 0), (xi, altura), VERMELHO, 2)
        # Rótulo à direita da linha; à esquerda, se não couber antes da borda. O
        # tamanho acompanha a altura do vídeo (1.0 em 1080 linhas).
        escala_rotulo = altura / 1080
        w = largura_texto(rotulo, escala_rotulo, 2)
        xt = xi + 8 if xi + 8 + w < largura else xi - 8 - w
        texto(frame, rotulo, (xt, int(altura * 0.6)), escala_rotulo, VERMELHO, 2)

    if np.isnan(dados["bbox"][i, 0]):
        return

    x1, y1, x2, y2 = dados["bbox"][i].astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), VERDE, 2)

    if not np.isnan(dados["cx"][i]):
        xe, ye, xd, yd = dados["ombros"][i]
        cv2.circle(frame, (int(xe), int(ye)), 4, AMARELO, -1)
        cv2.circle(frame, (int(xd), int(yd)), 4, AMARELO, -1)
        cx = dados["cx"][i]
        cy = (ye + yd) / 2
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, AZUL, -1)
        cv2.line(frame, (int(round(cx)), int(cy) - 25), (int(round(cx)), int(cy) + 25), AZUL, 1)


def desenhar_lupa(tela, frame, dados, i, escala):
    """
    Recorte ampliado em volta do corredor. Fica no canto superior direito, ou ao
    lado do painel quando o corredor está na direita (para não cobri-lo).
    """
    if np.isnan(dados["bbox"][i, 0]):
        return

    x1, y1, x2, y2 = dados["bbox"][i]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    altura, largura = frame.shape[:2]
    meio = LADO_LUPA_PX // 2

    x0 = int(min(max(cx - meio, 0), largura - LADO_LUPA_PX))
    y0 = int(min(max(cy - meio, 0), altura - LADO_LUPA_PX))
    recorte = frame[y0:y0 + LADO_LUPA_PX, x0:x0 + LADO_LUPA_PX]

    lado = int(LADO_LUPA_PX * FATOR_LUPA)
    lupa = cv2.resize(recorte, (lado, lado), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(lupa, (0, 0), (lado - 1, lado - 1), BRANCO, 2)

    canto_direito = tela.shape[1] - lado - 10
    if cx * escala > canto_direito - 220:
        xt = 440
    else:
        xt = canto_direito
    tela[10:10 + lado, xt:xt + lado] = lupa


def desenhar_painel(tela, estado, t_trecho, v_bruta, kalman_agora, i, t_video, status):
    painel(tela, 10, 10, 430, 205)

    cores = {"AGUARDANDO ENTRADA": AMARELO, "NO TRECHO": VERDE, "FINALIZADO": VERMELHO}
    texto(tela, estado, (22, 42), 0.85, cores[estado], 2)

    texto(tela, f"Tempo no trecho: {fmt(t_trecho)} s" if t_trecho is not None
          else "Tempo no trecho: -", (22, 75))
    texto(tela, f"V instantânea (bruta): {fmt(v_bruta)} m/s" if v_bruta is not None
          else "V instantânea (bruta): -", (22, 103), cor=CINZA)

    if kalman_agora is not None:
        v, dp = kalman_agora
        texto(tela, f"V com filtro: {fmt(v)} ± {fmt(dp)} m/s", (22, 131), 0.65, CIANO, 2)
    else:
        texto(tela, "V com filtro: -", (22, 131), 0.65, CIANO, 2)

    rodape = f"quadro {i} | t {fmt(t_video, 2)} s" + (f" | {status}" if status else "")
    texto(tela, rodape, (22, 192), 0.5, CINZA)


def desenhar_grafico(tela, medicao, ate_t, finalizado):
    """
    V instantânea ao longo do trecho. Eixo x: tempo desde a entrada.
    Pontos cinza: V bruta; vermelhos: pares que usam medição rejeitada pelo Kalman.
    Ciano: Kalman suavizado, com faixa ±2σ (covariância suavizada); círculos brancos
    nos extremos da curva suavizada no trecho, lidos de medir().
    """
    altura, largura = tela.shape[:2]
    gy1 = altura - 32
    gy0 = int(altura * (1 - ALTURA_GRAFICO))
    gx0, gx1 = 60, largura - 20

    painel(tela, 0, gy0 - 30, largura, altura, 0.6)

    t_total = medicao["t_trecho"]
    t0 = medicao["entrada"]["t"]
    k = medicao.get("kalman")

    # Escala vertical só de exibição: até 1,5 m/s acima da V máxima do Kalman
    # (entre 4 m/s e V_MAX_GRAFICO); valores acima ficam no topo.
    topo = V_MAX_GRAFICO
    if k is not None and "v_max_rts" in k:
        topo = float(min(V_MAX_GRAFICO, max(4.0, np.ceil(k["v_max_rts"] + 1.5))))
    passo_eixo = 1 if topo <= 6 else 2

    def px(t, v):
        x = gx0 + (t - t0) / t_total * (gx1 - gx0)
        y = gy1 - min(max(v, 0), topo) / topo * (gy1 - gy0)
        return int(round(x)), int(round(y))

    for v in range(0, int(topo) + 1, passo_eixo):
        _, y = px(t0, v)
        cv2.line(tela, (gx0, y), (gx1, y), (70, 70, 70), 1)
        texto(tela, str(v), (gx0 - 30, y + 5), 0.45, CINZA)

    texto(tela, "m/s", (8, gy0 - 8), 0.45, CINZA)
    texto(tela, "0 s", (gx0 - 5, altura - 10), 0.45, CINZA)
    texto(tela, f"{fmt(t_total, 2)} s", (gx1 - 45, altura - 10), 0.45, CINZA)

    legenda = "cinza: V bruta | vermelho: rejeitado"
    if k is not None:
        legenda += " | ciano: Kalman suavizado ±2σ | branco: mínimo e máximo"
    texto(tela, legenda, (gx0 + 60, altura - 10), 0.42, CINZA)

    rejeitados = set(k["frames_rejeitados"]) if k is not None else set()

    for a in medicao["instantaneas"]:
        if a["t"] > ate_t:
            break
        suspeito = a["frame"] in rejeitados or (a["frame"] - 1) in rejeitados
        cv2.circle(tela, px(a["t"], a["v_projetada"]), 3 if suspeito else 2,
                   VERMELHO if suspeito else CINZA, -1)

    if k is not None:
        sel = (k["t"] >= t0) & (k["t"] <= ate_t)

        if sel.sum() > 1:
            t_s = k["t"][sel]
            v_s = k["v_rts"][sel]
            dp_s = k["dp_v_rts"][sel]

            faixa = [px(t, v + 2 * d) for t, v, d in zip(t_s, v_s, dp_s)]
            faixa += [px(t, v - 2 * d) for t, v, d in zip(t_s[::-1], v_s[::-1], dp_s[::-1])]
            sobreposicao = tela.copy()
            cv2.fillPoly(sobreposicao, [np.array(faixa, np.int32)], (120, 120, 0))
            cv2.addWeighted(sobreposicao, 0.45, tela, 0.55, 0, tela)

            pontos = [px(t, v) for t, v in zip(t_s, v_s)]
            cv2.polylines(tela, [np.array(pontos, np.int32)], False, CIANO, 2, cv2.LINE_AA)

        for nome, rotulo in (("min", "mín"), ("max", "máx")):
            if f"v_{nome}_rts" not in k:
                continue
            t_ext = k["t_entrada_rts"] + k[f"t_v_{nome}_rts"]
            if t_ext <= ate_t:
                p = px(t_ext, k[f"v_{nome}_rts"])
                cv2.circle(tela, p, 6, BRANCO, 2, cv2.LINE_AA)
                texto(tela, rotulo, (p[0] + 9, p[1] + 5), 0.45, BRANCO)

    if finalizado:
        # Linhas de referência: velocidade média no trecho (distância / tempo) com o
        # tempo do cronômetro e com o da visão computacional.
        referencias = [(medicao["v_media"], AMARELO, "V média: visão computacional")]
        if "v_ref" in medicao:
            referencias.append((medicao["v_ref"], BRANCO, "V média: cronômetro"))

        for j, (v, cor, rotulo) in enumerate(referencias):
            _, y = px(t0, v)
            cv2.line(tela, (gx0, y), (gx1, y), cor, 1, cv2.LINE_AA)
            texto(tela, rotulo, (gx1 - 470 + 250 * j, gy0 - 8), 0.45, cor)


def desenhar_resultado(tela, medicao):
    """
    Velocidade média no trecho (distância / tempo) com o tempo de três origens:
    cronômetro, visão computacional sem filtro (cruzamentos interpolados) e com filtro
    (cruzamentos da posição suavizada pelo Kalman). Depois, os extremos da velocidade
    suavizada no trecho.
    """
    largura = tela.shape[1]
    k = medicao.get("kalman")
    tem_kalman = k is not None and "v_media_rts" in k

    linhas = []
    if "v_ref" in medicao:
        linhas.append(("Cronômetro", f"{fmt(medicao['t_cronometro'], 2)} s",
                       f"{fmt(medicao['v_ref'], 4)} m/s", BRANCO))
    linhas.append(("Visão computacional, sem filtro", f"{fmt(medicao['t_trecho'], 4)} s",
                   f"{fmt(medicao['v_media'], 4)} m/s", AMARELO))
    if tem_kalman:
        linhas.append(("Visão computacional, com filtro", f"{fmt(k['t_trecho_rts'], 4)} s",
                       f"{fmt(k['v_media_rts'], 4)} m/s", CIANO))

    extremos = []
    if tem_kalman and "v_min_rts" in k:
        for rotulo, nome in (("V mínima", "min"), ("V máxima", "max")):
            extremos.append(f"{rotulo}, com filtro: {fmt(k[f'v_{nome}_rts'], 2)} ± "
                            f"{fmt(k[f'dp_v_{nome}_rts'], 2)} m/s em t = {fmt(k[f't_v_{nome}_rts'], 2)} s")

    passo, largura_painel = 30, 620
    x0, y0 = largura // 2 - largura_painel // 2, 215
    altura_painel = 38 + passo + 6 + passo * len(linhas) + (12 + passo * len(extremos) if extremos else 0) + 18
    painel(tela, x0, y0, x0 + largura_painel, y0 + altura_painel, 0.72)

    y = y0 + 38
    texto(tela, f"RESULTADO — trecho de {fmt(medicao['distancia'], 0)} m", (x0 + 18, y), 0.8, VERMELHO, 2)
    col_t, col_v = x0 + 335, x0 + 455
    y += passo + 6
    texto(tela, "Tempo", (col_t, y), 0.5, CINZA)
    texto(tela, "V média no trecho", (col_v, y), 0.5, CINZA)

    for rotulo, t_txt, v_txt, cor in linhas:
        y += passo
        texto(tela, rotulo, (x0 + 18, y), 0.58, cor)
        texto(tela, t_txt, (col_t, y), 0.58, cor)
        texto(tela, v_txt, (col_v, y), 0.58, cor, 2)

    if extremos:
        y += 12
    for linha in extremos:
        y += passo
        texto(tela, linha, (x0 + 18, y), 0.58, CIANO)


# =============================================================================
# VALORES NO INSTANTE ATUAL (só leitura do que medir() calculou)
# =============================================================================

def kalman_no_instante(medicao, t):
    """(V suavizada, σ) do Kalman no último frame ≤ t."""
    k = medicao.get("kalman")
    if k is None:
        return None

    i = np.searchsorted(k["t"], t, side="right") - 1
    if i < 0:
        return None

    return k["v_rts"][i], k["dp_v_rts"][i]


def v_inst_no_frame(amostras, t):
    """Última V instantânea bruta disponível até o instante t."""
    anteriores = [a for a in amostras if a["t"] <= t]
    return anteriores[-1]["v_projetada"] if anteriores else None


# =============================================================================
# PRINCIPAL
# =============================================================================

def renderizar(frame, i, dados, medicao, escala, status):
    """Monta a tela do frame i. Trabalha numa cópia; o frame original não muda."""
    frame = frame.copy()
    amostras = medicao["instantaneas"]
    t_entrada = medicao["entrada"]["t"]
    t_saida = medicao["saida"]["t"]
    t_video = dados["t"][i]

    if t_video < t_entrada:
        estado, t_trecho = "AGUARDANDO ENTRADA", None
    elif t_video < t_saida:
        estado, t_trecho = "NO TRECHO", t_video - t_entrada
    else:
        estado, t_trecho = "FINALIZADO", medicao["t_trecho"]

    no_trecho = estado != "AGUARDANDO ENTRADA"
    ate_t = min(t_video, t_saida)

    desenhar_no_original(frame, dados, medicao, i)
    tela = reduzir_para_exibicao(frame, escala).copy()
    desenhar_lupa(tela, frame, dados, i, escala)
    desenhar_grafico(tela, medicao, ate_t if no_trecho else -1.0, estado == "FINALIZADO")
    desenhar_painel(
        tela, estado, t_trecho,
        v_inst_no_frame(amostras, ate_t) if no_trecho else None,
        kalman_no_instante(medicao, ate_t) if no_trecho else None,
        i, t_video, status,
    )

    if estado == "FINALIZADO":
        desenhar_resultado(tela, medicao)

    return tela


def gravar_tela(frame, i, dados, medicao, nome_arquivo):
    """Renderiza o quadro i na largura da figura e grava PNG em monografia/figuras."""
    escala = min(1.0, LARGURA_FIGURA / frame.shape[1])
    tela = renderizar(frame, i, dados, medicao, escala, status="")
    destino = PASTA_PROJETO / PASTA_FIGURAS / nome_arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    ok, png = cv2.imencode(".png", tela)
    if not ok:
        raise SystemExit(f"Falha ao codificar {nome_arquivo}")
    destino.write_bytes(png.tobytes())
    print(f"gravada: {destino.relative_to(PASTA_PROJETO)} (quadro {i}, {tela.shape[1]}x{tela.shape[0]})")


def salvar_quadro(cap, dados, medicao, quadro, nome_video):
    """
    Lê o vídeo em sequência até o quadro pedido (o último, se quadro < 0), sem
    abrir janela, e grava a tela. Leitura sequencial, como na reprodução: o índice
    do quadro é o mesmo da linha do CSV.
    """
    n_frames = len(dados["frame"])
    alvo = n_frames - 1 if quadro < 0 else quadro
    if not 0 <= alvo < n_frames:
        raise SystemExit(f"Quadro {quadro} fora do vídeo (0 a {n_frames - 1}).")

    frame = None
    for i in range(alvo + 1):
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"O vídeo acabou no quadro {i}, antes do {alvo}.")

    nome = f"tela_{nome_video}.png" if quadro < 0 else f"tela_{nome_video}_q{alvo}.png"
    gravar_tela(frame, alvo, dados, medicao, nome)


def visualizar(caminho_keypoints, caminho_calibracao, t_cronometro, velocidade, quadro_salvar=None):
    caminho_keypoints = resolver(caminho_keypoints)
    dados = carregar_keypoints(caminho_keypoints)
    meta = dados["meta"]

    nome_video = meta.get("video")
    if nome_video is None:
        raise SystemExit("O CSV não tem JSON de metadados ao lado; não sei qual é o vídeo.")

    if caminho_calibracao is None:
        caminho_calibracao = f"calibracao/{Path(nome_video).stem}.json"
    calibracao = json.loads(resolver(caminho_calibracao).read_text(encoding="utf-8"))

    # Relatório completo no terminal: o mesmo do analisar_cruzamentos.py.
    medicao = analisar(caminho_keypoints, calibracao, t_cronometro)

    if medicao["entrada"] is None or medicao["saida"] is None:
        raise SystemExit("Sem cruzamentos de entrada e saída; nada para visualizar.")

    cap = cv2.VideoCapture(str(resolver(f"videos/{nome_video}")))
    if not cap.isOpened():
        raise SystemExit(f"Não abri videos/{nome_video}")

    if quadro_salvar is not None:
        salvar_quadro(cap, dados, medicao, quadro_salvar, Path(nome_video).stem)
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala = calcular_escala_exibicao(largura, altura)
    n_frames = len(dados["frame"])

    print("\nESPAÇO = pausar | + / - = velocidade | R = recomeçar | S = gravar a tela | Q = sair")

    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_AUTOSIZE)

    i = -1
    frame = None
    pausado = False
    fim = False

    while True:
        inicio = time.perf_counter()

        if not pausado and not fim:
            ok, novo = cap.read()

            if ok and i + 1 < n_frames:
                frame = novo
                i += 1
            else:
                # Fim do vídeo: fica no último quadro, com o resultado.
                fim = True

        if frame is None:
            break

        if fim:
            status = "FIM: R recomeça, S grava, Q sai"
        elif pausado:
            status = "PAUSADO"
        else:
            status = f"{velocidade}x"

        cv2.imshow(NOME_JANELA, renderizar(frame, i, dados, medicao, escala, status))

        periodo_ms = 1000.0 / fps / velocidade
        gasto_ms = (time.perf_counter() - inicio) * 1000
        tecla = cv2.waitKey(max(1, int(periodo_ms - gasto_ms))) & 0xFF

        if cv2.getWindowProperty(NOME_JANELA, cv2.WND_PROP_VISIBLE) < 1:
            break

        if tecla == ord("q"):
            break

        elif tecla == ord(" "):
            pausado = not pausado

        elif tecla == ord("s"):
            gravar_tela(frame, i, dados, medicao, f"tela_{Path(nome_video).stem}_q{i}.png")

        elif tecla == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            i = -1
            pausado = False
            fim = False

        elif tecla in (ord("+"), ord("=")):
            k = VELOCIDADES_DISPONIVEIS.index(velocidade) if velocidade in VELOCIDADES_DISPONIVEIS else 2
            velocidade = VELOCIDADES_DISPONIVEIS[min(k + 1, len(VELOCIDADES_DISPONIVEIS) - 1)]

        elif tecla in (ord("-"), ord("_")):
            k = VELOCIDADES_DISPONIVEIS.index(velocidade) if velocidade in VELOCIDADES_DISPONIVEIS else 2
            velocidade = VELOCIDADES_DISPONIVEIS[max(k - 1, 0)]

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# ESCOLHA DO VÍDEO
# =============================================================================

def keypoints_disponiveis():
    """
    {nome do vídeo: CSV de keypoints}. Para cada vídeo usa a extração de maior
    imgsz (a resolução nativa, padrão do pipeline).
    """
    melhores = {}

    for csv_kp in sorted((PASTA_PROJETO / PASTA_KEYPOINTS).glob("*_imgsz*.csv")):
        nome, _, imgsz = csv_kp.stem.rpartition("_imgsz")
        if not imgsz.isdigit():
            continue
        if nome not in melhores or int(imgsz) > melhores[nome][0]:
            melhores[nome] = (int(imgsz), csv_kp)

    return {nome: caminho for nome, (_, caminho) in melhores.items()}


def escolher_video(disponiveis):
    """Menu no terminal, para rodar sem argumentos (ex.: botão Run do VS Code)."""
    nomes = sorted(disponiveis)

    if not nomes:
        raise SystemExit("Nenhum CSV em resultados/keypoints. Rode src/extrair_keypoints.py antes.")

    print("\nVídeos com keypoints extraídos:")
    for k, nome in enumerate(nomes, start=1):
        print(f"  {k}) {nome}   ({disponiveis[nome].name})")

    while True:
        resposta = input("Escolha o número do vídeo: ").strip()
        if resposta.isdigit() and 1 <= int(resposta) <= len(nomes):
            return nomes[int(resposta) - 1]
        print("Opção inválida.")


def cronometro_registrado(nome_video):
    """Último t_cronometro de resultados/ensaios.csv para o vídeo, se houver."""
    caminho = PASTA_PROJETO / ENSAIOS
    if not caminho.exists():
        return None

    with open(caminho, encoding="utf-8") as arquivo:
        linhas = [l for l in csv.DictReader(arquivo)
                  if Path(l["video"]).stem == nome_video and l["t_cronometro"]]

    return float(linhas[-1]["t_cronometro"]) if linhas else None


def main():
    parser = argparse.ArgumentParser(description="Reproduz o vídeo com a medição sobreposta.")
    parser.add_argument("--video", default=None,
                        help="nome do vídeo (ex.: J_Variando). Sem isso, abre um menu.")
    parser.add_argument("--keypoints", default=None,
                        help="CSV específico (sobrepõe --video)")
    parser.add_argument("--calibracao", default=None,
                        help="padrão: calibracao/<video>.json")
    parser.add_argument("--t-cronometro", type=float, default=None,
                        help="padrão: o registrado em resultados/ensaios.csv")
    parser.add_argument("--velocidade", type=float, default=VELOCIDADE_REPRODUCAO,
                        help="velocidade de reprodução (1.0 = tempo real)")
    parser.add_argument("--salvar-quadro", type=int, nargs="?", const=-1, default=None, metavar="N",
                        help="não abre janela: grava a tela do quadro N (sem N, o último, com o "
                             f"resultado) em PNG em {PASTA_FIGURAS}/")
    args = parser.parse_args()

    if args.keypoints is not None:
        caminho_keypoints = args.keypoints
        nome_video = Path(args.keypoints).stem.rpartition("_imgsz")[0]
    else:
        disponiveis = keypoints_disponiveis()
        nome_video = Path(args.video).stem if args.video else escolher_video(disponiveis)
        if nome_video not in disponiveis:
            raise SystemExit(f"Sem keypoints para {nome_video}. Disponíveis: {sorted(disponiveis)}")
        caminho_keypoints = disponiveis[nome_video]

    t_cronometro = args.t_cronometro
    if t_cronometro is None:
        t_cronometro = cronometro_registrado(nome_video)
        if t_cronometro is not None:
            print(f"Cronômetro de {ENSAIOS}: {t_cronometro} s")

    visualizar(caminho_keypoints, args.calibracao, t_cronometro, args.velocidade, args.salvar_quadro)


if __name__ == "__main__":
    main()
