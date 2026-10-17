"""
=============================================================================
ETAPA 2 DO PIPELINE — CRUZAMENTOS E VELOCIDADE MÉDIA
(CSV de keypoints + calibração → tempos, velocidade, diagnóstico)
=============================================================================

Lê o CSV gerado por src/extrair_keypoints.py (ou um backlog legado de
src/Testar_Keypoints_YOLO.py) e reproduz, sem rodar o YOLO, a lógica de
cruzamento do legado src/Medir_Velocidade_Ombros_Interpolado.py:

  - ponto medido: centro dos ombros (COCO 5 e 6), os dois com conf ≥ 0,35;
  - frames sem medição válida não são preenchidos: o intervalo simplesmente cresce;
  - entrada: medição anterior fora da zona e atual dentro dela;
  - cruzamento interpolado linearmente entre as duas medições válidas que cercam
    a linha, com 0 ≤ alfa ≤ 1;
  - velocidade média no trecho = distância / (t_saída − t_entrada).

O que este arquivo acrescenta ao legado:
  - CRITÉRIO DE GAP MÁXIMO no cruzamento (GAP_MAX_CRUZAMENTO_S), que só sinaliza:
    o valor continua no relatório, marcado como inválido;
  - base de tempo pelo carimbo de cada quadro, e não por quadro / FPS do contêiner
    (desde 2026-10-17; ver carregar_keypoints). O tempo nominal fica no relatório
    como referência;
  - velocidade instantânea em duas versões lado a lado: com sinal, projetada na
    direção da passagem, e com |Δx| como no legado (problema conhecido 2 do README);
  - comparação com o cronômetro;
  - vários CSVs de uma vez, com tabela lado a lado no final.

A V instantânea é a BRUTA, par a par. Nenhum filtro é aplicado: decisão do autor
em 2026-10-07 (problema conhecido 4 do README).
O visualizador (src/visualizar_medicao.py) usa a função medir() deste arquivo,
então os números da tela e os do relatório vêm do mesmo cálculo.

Uso (a partir da pasta tcc-velocidade):

    python src/analisar_cruzamentos.py resultados/keypoints/J_Correndo_imgsz640.csv \\
        --calibracao calibracao/J_Correndo.json --t-cronometro 3.48
=============================================================================
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import kalman


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_PROJETO = Path(__file__).resolve().parent.parent

# COCO Pose: 5 = ombro esquerdo, 6 = ombro direito.
INDICE_OMBRO_ESQUERDO = 5
INDICE_OMBRO_DIREITO = 6

# Os dois ombros precisam satisfazer esse limiar (igual ao legado).
CONFIANCA_MIN_OMBRO = 0.35

# -----------------------------------------------------------------------------
# GAP MÁXIMO PARA INTERPOLAR UM CRUZAMENTO
#
# Critério definido em 2026-10-07, antes de rodar as variantes de imgsz, e
# aplicado igualmente a todos os ensaios. Cruzamento com gap acima do limite NÃO
# é descartado: o valor continua no relatório, marcado como inválido.
#
# Justificativa (orçamento de erro): a interpolação linear entre duas medições
# separadas por Δt erra a posição em até a·Δt²/8 se o corredor acelera com a.
# Em tempo, δt ≈ a·Δt² / (8·v). Com a = 5 m/s² (conservador, saída de sprint) e
# v = 5 m/s, Δt = 0,25 s dá δt ≈ 8 ms por cruzamento: menos de meio período de
# quadro a 30 fps e ~0,2% de um trecho de 3,5 s. 0,25 s é também ~1 passada a
# 4 Hz. Acima disso o erro cresce com Δt² e, como no J_Correndo com imgsz 640
# (gap de 3,15 s), o par de medições pode nem pertencer ao mesmo movimento.
#
# Valor provisório: discutir com o orientador.
# -----------------------------------------------------------------------------
GAP_MAX_CRUZAMENTO_S = 0.25

# Kalman (src/kalman.py): janela além dos cruzamentos, para o suavizador RTS ter
# medições dos dois lados de cada linha.
MARGEM_KALMAN_S = 0.5

# Ruído de medição R do Kalman:
#   "verossimilhanca" — R por máxima verossimilhança, partindo do σx do ensaio
#                       (padrão desde 2026-10-07; ver experimentos/2026-10-07-kalman.md)
#   "sigma_x"         — R = (σx · escala)², σx da segunda diferença (versão 1:
#                       subestima R e faz o teste de inovação rejeitar medição boa)
R_KALMAN = "verossimilhanca"
SIGMA_X_PADRAO_PX = 1.0   # se o ensaio não tiver σx


# =============================================================================
# LEITURA
# =============================================================================

def resolver(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho


def carregar_keypoints(caminho):
    """
    Lê o CSV por nome de coluna, para aceitar tanto o formato novo quanto o
    backlog legado (que só tem timestamp_s nominal e não tem n_pessoas).
    """
    with open(caminho, encoding="utf-8-sig") as arquivo:
        linhas = list(csv.reader(arquivo, delimiter=";"))

    cabecalho = linhas[0]
    col = {nome: i for i, nome in enumerate(cabecalho)}

    def coluna(nome):
        i = col[nome]
        return np.array([float(l[i]) if l[i] != "" else np.nan for l in linhas[1:]])

    def coluna_kp(indice, eixo):
        nome = next(n for n in cabecalho
                    if n.startswith(f"kp{indice}_") and n.endswith(f"_{eixo}"))
        return coluna(nome)

    dados = {
        "frame": coluna("frame").astype(int),
        "t_nominal": coluna("t_nominal_s" if "t_nominal_s" in col else "timestamp_s"),
        "t_carimbo": coluna("t_ms") / 1000.0 if "t_ms" in col else None,
        "n_pessoas": coluna("n_pessoas") if "n_pessoas" in col else None,
        "pessoa": coluna("pessoa_detectada"),
    }

    # Base de tempo oficial: o carimbo de cada quadro (CAP_PROP_POS_MSEC), desde
    # 2026-10-17. O tempo nominal divide o índice do quadro pela taxa média do
    # contêiner, que nos vídeos a 60 fps vem distorcida pelo primeiro intervalo
    # anômalo (59,979 e 60,077 fps; os carimbos dão 60). Ele fica só como referência
    # no relatório e como reserva para o backlog legado, que não tem carimbo.
    tem_carimbo = dados["t_carimbo"] is not None
    dados["t"] = dados["t_carimbo"] if tem_carimbo else dados["t_nominal"]
    dados["base_tempo"] = "carimbo" if tem_carimbo else "nominal"

    xe = coluna_kp(INDICE_OMBRO_ESQUERDO, "x")
    ye = coluna_kp(INDICE_OMBRO_ESQUERDO, "y")
    ce = coluna_kp(INDICE_OMBRO_ESQUERDO, "conf")
    xd = coluna_kp(INDICE_OMBRO_DIREITO, "x")
    yd = coluna_kp(INDICE_OMBRO_DIREITO, "y")
    cd = coluna_kp(INDICE_OMBRO_DIREITO, "conf")

    # Mesmo critério de validade do legado: coordenadas positivas e os dois
    # ombros com confiança mínima. NaN (sem pessoa) falha nas comparações.
    with np.errstate(invalid="ignore"):
        valido = (
            (xe > 0) & (ye > 0) & (xd > 0) & (yd > 0)
            & (ce >= CONFIANCA_MIN_OMBRO) & (cd >= CONFIANCA_MIN_OMBRO)
        )

    dados["cx"] = np.where(valido, (xe + xd) / 2.0, np.nan)

    # Guardados só para desenho (visualizador); não entram em nenhum cálculo.
    dados["ombros"] = np.column_stack([xe, ye, xd, yd])
    dados["bbox"] = np.column_stack([coluna(c) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")])

    caminho_json = Path(caminho).with_suffix(".json")
    dados["meta"] = (
        json.loads(caminho_json.read_text(encoding="utf-8"))
        if caminho_json.exists() else {}
    )

    return dados


# =============================================================================
# CRUZAMENTOS
# =============================================================================

def interpolar_cruzamento(dados, i_antes, i_depois, x_linha):
    """Mesma interpolação do legado, na base oficial e, como referência, na nominal."""
    x0 = dados["cx"][i_antes]
    x1 = dados["cx"][i_depois]

    if abs(x1 - x0) < 1e-12:
        return None

    alfa = (x_linha - x0) / (x1 - x0)

    if alfa < 0.0 or alfa > 1.0:
        return None

    def no_tempo(t):
        return t[i_antes] + alfa * (t[i_depois] - t[i_antes])

    t = dados["t"]
    gap_s = t[i_depois] - t[i_antes]

    return {
        "x_linha": x_linha,
        "frame_antes": int(dados["frame"][i_antes]),
        "frame_depois": int(dados["frame"][i_depois]),
        "gap_frames": int(dados["frame"][i_depois] - dados["frame"][i_antes]),
        "gap_s": gap_s,
        "gap_ok": gap_s <= GAP_MAX_CRUZAMENTO_S,
        "alfa": alfa,
        "t": no_tempo(t),
        "t_nominal": no_tempo(dados["t_nominal"]),
    }


def detectar_cruzamentos(dados, x_min, x_max):
    """
    Percorre as medições válidas na ordem dos frames, como a máquina de estados
    do legado: primeiro a entrada (define a direção), depois a saída.
    """
    validos = np.flatnonzero(~np.isnan(dados["cx"]))
    cx = dados["cx"]

    direcao = None
    entrada = None
    x_saida = None
    anterior = None

    for i in validos:
        if anterior is None:
            anterior = i
            continue

        x_ant = cx[anterior]
        x_atual = cx[i]

        if entrada is None:
            dentro = x_min <= x_atual <= x_max

            if x_ant < x_min and dentro:
                direcao, x_entrada, x_saida = 1, x_min, x_max
            elif x_ant > x_max and dentro:
                direcao, x_entrada, x_saida = -1, x_max, x_min
            else:
                anterior = i
                continue

            entrada = interpolar_cruzamento(dados, anterior, i, x_entrada)

            if entrada is None:
                direcao = None

        else:
            if direcao == 1:
                saiu = x_ant < x_saida and x_atual >= x_saida
            else:
                saiu = x_ant > x_saida and x_atual <= x_saida

            if saiu:
                saida = interpolar_cruzamento(dados, anterior, i, x_saida)
                if saida is not None:
                    return direcao, entrada, saida

        anterior = i

    return direcao, entrada, None


# =============================================================================
# DIAGNÓSTICO DO TRECHO
# =============================================================================

def ruido_segunda_diferenca(serie, frames):
    """
    Ruído de posição por segunda diferença em trincas de frames consecutivos:
    para movimento ~uniforme, x[t+1] − 2x[t] + x[t−1] tem desvio √6·σ.
    Estimador robusto (MAD). Inclui também aceleração real e oscilação do
    tronco, então é um limite superior para o ruído do keypoint.
    """
    d2 = []

    for k in range(1, len(frames) - 1):
        consecutivos = frames[k] - frames[k - 1] == 1 and frames[k + 1] - frames[k] == 1
        if consecutivos and not np.isnan(serie[k - 1:k + 2]).any():
            d2.append(serie[k + 1] - 2 * serie[k] + serie[k - 1])

    if len(d2) < 5:
        return np.nan, len(d2)

    d2 = np.array(d2)
    mad = np.median(np.abs(d2 - np.median(d2))) * 1.4826

    return mad / np.sqrt(6), len(d2)


def diagnosticar_trecho(dados, entrada, saida):
    """
    Trecho = do frame que confirmou a entrada ao frame que confirmou a saída,
    inclusive (mesma contagem do relatório legado).
    """
    frames = dados["frame"]
    no_trecho = (frames >= entrada["frame_depois"]) & (frames <= saida["frame_depois"])
    validos = no_trecho & ~np.isnan(dados["cx"])

    t_validos = dados["t"][validos]
    maior_gap_s = float(np.max(np.diff(t_validos))) if validos.sum() > 1 else np.nan

    janela = (frames >= entrada["frame_antes"]) & (frames <= saida["frame_depois"])
    sigma_x, n_trincas = ruido_segunda_diferenca(
        np.where(janela, dados["cx"], np.nan), frames
    )

    if dados["n_pessoas"] is not None:
        multiplas = int(np.sum(no_trecho & (dados["n_pessoas"] > 1)))
    else:
        multiplas = None

    return {
        "frames_trecho": int(no_trecho.sum()),
        "validos_trecho": int(validos.sum()),
        "disponibilidade": 100.0 * validos.sum() / max(no_trecho.sum(), 1),
        "maior_gap_s": maior_gap_s,
        "sigma_x_px": sigma_x,
        "n_trincas": n_trincas,
        "frames_multiplas_pessoas": multiplas,
    }


# =============================================================================
# VELOCIDADE INSTANTÂNEA
# =============================================================================

def calcular_velocidades_instantaneas(dados, entrada, saida, direcao, metros_por_pixel):
    """
    Uma velocidade por par de medições válidas consecutivas dentro do trecho,
    com os mesmos pares do legado: do frame que confirmou a entrada até a última
    medição antes da linha de saída (o par que cruza a saída não entra). Frames
    sem medição não são preenchidos: o Δt do par simplesmente cresce.

    Duas versões, guardadas lado a lado durante a transição (problema conhecido 2 do README):
      v_projetada = direção · Δx · escala / Δt   (com sinal; negativa = suspeita)
      v_modulo    = |Δx| · escala / Δt           (como o legado)
    A diferença entre as médias das duas é o viés de retificação do abs().
    Negativas NÃO são descartadas: entram na média e são contadas no relatório.
    """
    frames = dados["frame"]
    cx = dados["cx"]
    t = dados["t"]

    indices = np.flatnonzero(
        (frames >= entrada["frame_depois"])
        & (frames <= saida["frame_antes"])
        & ~np.isnan(cx)
    )

    amostras = []

    for a, b in zip(indices[:-1], indices[1:]):
        delta_t = t[b] - t[a]
        delta_x = cx[b] - cx[a]

        amostras.append({
            "frame": int(frames[b]),
            "t": float(t[b]),
            "delta_t": float(delta_t),
            "v_projetada": direcao * delta_x * metros_por_pixel / delta_t,
            "v_modulo": abs(delta_x) * metros_por_pixel / delta_t,
        })

    return amostras


def resumir_instantaneas(amostras):
    if not amostras:
        return None

    v_proj = np.array([a["v_projetada"] for a in amostras])
    v_mod = np.array([a["v_modulo"] for a in amostras])

    return {
        "n": len(amostras),
        "media_projetada": float(v_proj.mean()),
        "media_modulo": float(v_mod.mean()),
        "desvio_projetada": float(v_proj.std(ddof=1)) if len(v_proj) > 1 else np.nan,
        "minimo_projetada": float(v_proj.min()),
        "maximo_projetada": float(v_proj.max()),
        "negativas": int(np.sum(v_proj < 0)),
    }


# =============================================================================
# KALMAN
# =============================================================================

def cruzamento_continuo(t, z, z_linha, depois_de=-np.inf):
    """Primeiro instante (após `depois_de`) em que a série z sobe através de z_linha."""
    for i in range(1, len(z)):
        if t[i] > depois_de and z[i - 1] < z_linha <= z[i]:
            alfa = (z_linha - z[i - 1]) / (z[i] - z[i - 1])
            return t[i - 1] + alfa * (t[i] - t[i - 1])
    return None


def aplicar_kalman(dados, m, modo_r=None):
    """
    Roda o filtro e o suavizador de Kalman (modelo CV, src/kalman.py) sobre a
    posição do centro dos ombros projetada na direção da corrida (m), numa janela
    em volta do trecho.

    Devolve um único dicionário: q e R por máxima verossimilhança, consistência,
    frames rejeitados pelo teste de inovação, séries de velocidade (filtro e
    suavizador, com desvio padrão) e a velocidade média no trecho pelo Kalman: os
    instantes em que a posição SUAVIZADA dos ombros cruza as duas linhas.
    """
    frames = dados["frame"]
    t = dados["t"]
    entrada, saida = m["entrada"], m["saida"]
    escala = m["metros_por_pixel"]

    janela = ((t >= entrada["t"] - MARGEM_KALMAN_S)
              & (t <= saida["t"] + MARGEM_KALMAN_S))
    t_j = t[janela]
    f_j = frames[janela]
    z_j = m["direcao"] * dados["cx"][janela] * escala

    sigma_x = m["sigma_x_px"] if np.isfinite(m["sigma_x_px"]) else SIGMA_X_PADRAO_PX
    r = (sigma_x * escala) ** 2
    modo_r = modo_r or R_KALMAN
    ajuste_r = {"fator": 1.0, "no_limite": False}

    if modo_r == "verossimilhanca":
        ajuste_r = kalman.ajustar_r(t_j, z_j, r)
        r = ajuste_r["r"]

    z_entrada = m["direcao"] * entrada["x_linha"] * escala
    z_saida = m["direcao"] * saida["x_linha"] * escala

    res = kalman.estimar(t_j, z_j, r)

    t_ent = cruzamento_continuo(t_j, res["z_rts"], z_entrada)
    t_sai = cruzamento_continuo(t_j, res["z_rts"], z_saida, t_ent if t_ent else -np.inf)

    res["frames"] = f_j
    res["t"] = t_j
    res["frames_rejeitados"] = f_j[res["rejeitado"]].tolist()

    if t_ent is not None and t_sai is not None:
        res["t_entrada_rts"] = t_ent
        res["t_saida_rts"] = t_sai
        res["t_trecho_rts"] = t_sai - t_ent
        res["v_media_rts"] = m["distancia"] / (t_sai - t_ent)

        no_trecho = (t_j >= t_ent) & (t_j <= t_sai)
        res["media_v_inst_rts"] = float(res["v_rts"][no_trecho].mean())
        res["dp_v_rts_trecho"] = float(res["v_rts"][no_trecho].std(ddof=1))
        res["dp_medio_v_rts"] = float(res["dp_v_rts"][no_trecho].mean())

        # Extremos da velocidade suavizada no trecho. É a velocidade do centro dos
        # ombros: inclui a oscilação da passada que o suavizador preserva (~90% a
        # 3,5 Hz), e o pico fica um pouco atenuado. Tempo contado desde a entrada.
        idx = np.flatnonzero(no_trecho)
        for nome, j in (("min", idx[np.argmin(res["v_rts"][idx])]),
                        ("max", idx[np.argmax(res["v_rts"][idx])])):
            res[f"v_{nome}_rts"] = float(res["v_rts"][j])
            res[f"t_v_{nome}_rts"] = float(t_j[j] - t_ent)
            res[f"dp_v_{nome}_rts"] = float(res["dp_v_rts"][j])
            res[f"frame_v_{nome}_rts"] = int(f_j[j])

    res.update({"r": r, "sigma_x_px": sigma_x, "modo_r": modo_r, "fator_r": ajuste_r["fator"],
                "r_no_limite": ajuste_r["no_limite"], "sigma_z_px": np.sqrt(r) / escala})
    return res


# =============================================================================
# MEDIÇÃO (sem impressão; usada pelo relatório e pelo visualizador)
# =============================================================================

def medir(dados, calibracao, t_cronometro=None, com_kalman=True):
    x_min, x_max = sorted(calibracao["x_linhas_px"])
    distancia = calibracao["distancia_real_m"]
    metros_por_pixel = distancia / (x_max - x_min)

    direcao, entrada, saida = detectar_cruzamentos(dados, x_min, x_max)

    m = {
        "x_min": x_min,
        "x_max": x_max,
        "distancia": distancia,
        "metros_por_pixel": metros_por_pixel,
        "direcao": direcao,
        "entrada": entrada,
        "saida": saida,
        "valido": False,
    }

    if entrada is None or saida is None:
        return m

    m.update(diagnosticar_trecho(dados, entrada, saida))

    # Velocidade média no trecho, na base oficial (carimbo); a nominal fica como
    # referência no relatório.
    m["base_tempo"] = dados["base_tempo"]
    m["t_trecho"] = saida["t"] - entrada["t"]
    m["v_media"] = distancia / m["t_trecho"]
    m["t_nominal"] = saida["t_nominal"] - entrada["t_nominal"]
    m["v_nominal"] = distancia / m["t_nominal"]

    m["instantaneas"] = calcular_velocidades_instantaneas(
        dados, entrada, saida, direcao, metros_por_pixel
    )
    m["resumo_inst"] = resumir_instantaneas(m["instantaneas"])

    if t_cronometro is not None:
        m["t_cronometro"] = t_cronometro
        m["v_ref"] = distancia / t_cronometro
        m["erro_v"] = m["v_media"] - m["v_ref"]
        m["erro_rel"] = 100.0 * m["erro_v"] / m["v_ref"]
        m["erro_t"] = m["t_trecho"] - t_cronometro

    m["valido"] = entrada["gap_ok"] and saida["gap_ok"]

    if com_kalman:
        m["kalman"] = aplicar_kalman(dados, m)

    return m


# =============================================================================
# RELATÓRIO
# =============================================================================

def analisar(caminho, calibracao, t_cronometro, com_kalman=True):
    dados = carregar_keypoints(caminho)
    meta = dados["meta"]
    m = medir(dados, calibracao, t_cronometro, com_kalman)

    print("\n" + "=" * 78)
    print(f"{Path(caminho).name}")
    if meta:
        print(f"imgsz {meta.get('imgsz')} | conf {meta.get('conf_deteccao')} | "
              f"ultralytics {meta.get('versoes', {}).get('ultralytics')} | "
              f"{meta.get('ms_por_frame')} ms/frame")
    print("=" * 78)
    print(f"Calibração: x {m['x_min']:.1f} e {m['x_max']:.1f} px | {m['distancia']:.3f} m | "
          f"{m['metros_por_pixel'] * 100:.4f} cm/px  [{calibracao.get('origem', '?')}]")
    print(f"Medições válidas (ombros ≥ {CONFIANCA_MIN_OMBRO}): "
          f"{int((~np.isnan(dados['cx'])).sum())} de {len(dados['frame'])} frames")

    resultado = dict(m)
    resultado.update({"arquivo": Path(caminho).name, "imgsz": meta.get("imgsz"),
                      "ms_por_frame": meta.get("ms_por_frame")})

    entrada, saida = m["entrada"], m["saida"]

    if entrada is None or saida is None:
        print("Cruzamento de entrada ou de saída NÃO encontrado.")
        return resultado

    print(f"Direção: {'esquerda -> direita' if m['direcao'] == 1 else 'direita -> esquerda'}")

    for rotulo, c in (("ENTRADA", entrada), ("SAÍDA", saida)):
        situacao = "OK" if c["gap_ok"] else "GAP ACIMA DO LIMITE"
        print(f"{rotulo:<8} x={c['x_linha']:.1f}: frames {c['frame_antes']} -> "
              f"{c['frame_depois']} (gap {c['gap_frames']} fr = {c['gap_s'] * 1000:.0f} ms, "
              f"alfa {c['alfa']:.3f}) -> t = {c['t']:.4f} s  [{situacao}]")

    print(f"Trecho: {m['frames_trecho']} frames, {m['validos_trecho']} válidos "
          f"({m['disponibilidade']:.1f}%) | maior gap interno "
          f"{m['maior_gap_s'] * 1000:.0f} ms")
    print(f"σx centro dos ombros: {m['sigma_x_px']:.2f} px "
          f"(≈ {m['sigma_x_px'] * m['metros_por_pixel'] * 100:.1f} cm, n={m['n_trincas']} trincas)"
          + (f" | frames com >1 pessoa: {m['frames_multiplas_pessoas']}"
             if m["frames_multiplas_pessoas"] is not None else ""))

    print(f"Tempo no trecho ({m['base_tempo']}): {m['t_trecho']:.4f} s | "
          f"V média no trecho {m['v_media']:.4f} m/s")

    if m["base_tempo"] == "carimbo":
        print(f"  referência, quadro / FPS do contêiner: {m['t_nominal']:.4f} s | "
              f"V {m['v_nominal']:.4f} m/s | carimbo − nominal: "
              f"{(m['t_trecho'] - m['t_nominal']) * 1000:+.2f} ms")

    r = m["resumo_inst"]
    if r is not None:
        print(f"V instantâneas: {r['n']} pares | média com sinal {r['media_projetada']:.4f} m/s | "
              f"média |Δx| (legado) {r['media_modulo']:.4f} m/s | "
              f"viés do abs {r['media_modulo'] - r['media_projetada']:+.4f} m/s | "
              f"pares negativos {r['negativas']}")
        print(f"                dp {r['desvio_projetada']:.3f} m/s | "
              f"min {r['minimo_projetada']:.3f} | max {r['maximo_projetada']:.3f} | "
              f"média com sinal − V média: {r['media_projetada'] - m['v_media']:+.4f} m/s "
              f"({100.0 * (r['media_projetada'] - m['v_media']) / m['v_media']:+.2f}%)")

    if "kalman" in m:
        imprimir_kalman(m)

    if "v_ref" in m:
        print(f"Cronômetro:              {m['t_cronometro']:.4f} s | V ref {m['v_ref']:.4f} m/s | "
              f"erro V {m['erro_v']:+.4f} m/s ({m['erro_rel']:+.2f}%) | "
              f"erro t {m['erro_t']:+.4f} s")

        k = m.get("kalman")
        if k is not None and "v_media_rts" in k:
            erro_k = k["v_media_rts"] - m["v_ref"]
            print(f"Cronômetro vs Kalman suavizado: erro V {erro_k:+.4f} m/s "
                  f"({100.0 * erro_k / m['v_ref']:+.2f}%) | "
                  f"erro t {k['t_trecho_rts'] - m['t_cronometro']:+.4f} s")

    if m["valido"]:
        print(f"STATUS: VÁLIDO (gaps ≤ {GAP_MAX_CRUZAMENTO_S * 1000:.0f} ms)")
    else:
        print(f"STATUS: INVÁLIDO — gap de cruzamento acima de "
              f"{GAP_MAX_CRUZAMENTO_S * 1000:.0f} ms. Valor reportado, não usar como medição.")

    return resultado


def imprimir_kalman(m):
    k = m["kalman"]
    origem_r = (f"máx. verossim. = {k['fator_r']:.2f} × σx de {k['sigma_x_px']:.2f} px"
                if k["modo_r"] == "verossimilhanca" else "σx da 2ª diferença")
    print(f"\nKALMAN — janela ±{MARGEM_KALMAN_S} s em volta do trecho | "
          f"R = ({k['sigma_z_px']:.2f} px)², {origem_r} | rejeição se NIS > {kalman.LIMIAR_INOVACAO:g}"
          + ("  AVISO: R na borda da grade" if k["r_no_limite"] else ""))
    print(f"  {'modelo':<8}{'q (m²/s³)':>10}{'NIS':>7}{'>χ²95':>8}{'lag1':>7}  "
          f"{'V média':>11}{'média V':>9}{'dp V':>7}{'±σ médio':>9}  rejeitados")

    c = k["consistencia"]
    v_media = f"{k['v_media_rts']:.4f}" if "v_media_rts" in k else "—"
    media = f"{k['media_v_inst_rts']:.4f}" if "media_v_inst_rts" in k else "—"
    dp = f"{k['dp_v_rts_trecho']:.3f}" if "dp_v_rts_trecho" in k else "—"
    sig = f"{k['dp_medio_v_rts']:.3f}" if "dp_medio_v_rts" in k else "—"
    print(f"  {k['modelo']:<8}{k['ajuste']['params']['q']:>10.4g}{c['nis_medio']:>7.3f}"
          f"{100 * c['fracao_acima_95']:>7.1f}%{c['autocorrelacao_lag1']:>+7.3f}  "
          f"{v_media:>11}{media:>9}{dp:>7}{sig:>9}  {k['frames_rejeitados']}")

    if k["ajuste"]["no_limite"]:
        print(f"  AVISO: ótimo na borda da grade de busca: {k['ajuste']['no_limite']}")

    print(f"  V média no trecho pela interpolação linear: {m['v_media']:.4f} m/s")

    if "v_min_rts" in k:
        print(f"  V suavizada no trecho: mín {k['v_min_rts']:.3f} ± {k['dp_v_min_rts']:.3f} m/s "
              f"(t = {k['t_v_min_rts']:.2f} s, frame {k['frame_v_min_rts']}) | "
              f"máx {k['v_max_rts']:.3f} ± {k['dp_v_max_rts']:.3f} m/s "
              f"(t = {k['t_v_max_rts']:.2f} s, frame {k['frame_v_max_rts']})")


def imprimir_tabela(resultados):
    print("\n" + "=" * 78)
    print("COMPARAÇÃO LADO A LADO")
    print("=" * 78)

    def fmt(valor, formato):
        return format(valor, formato) if valor is not None else "—"

    linhas = [
        ("imgsz", lambda r: fmt(r.get("imgsz"), "")),
        ("ms/frame", lambda r: fmt(r.get("ms_por_frame"), "")),
        ("gap entrada (ms)", lambda r: fmt(r["entrada"]["gap_s"] * 1000, ".0f") if r.get("entrada") else "—"),
        ("gap saída (ms)", lambda r: fmt(r["saida"]["gap_s"] * 1000, ".0f") if r.get("saida") else "—"),
        ("disponibilidade (%)", lambda r: fmt(r.get("disponibilidade"), ".1f")),
        ("maior gap interno (ms)", lambda r: fmt(r["maior_gap_s"] * 1000, ".0f") if "maior_gap_s" in r else "—"),
        ("σx ombros (px)", lambda r: fmt(r.get("sigma_x_px"), ".2f")),
        ("frames >1 pessoa", lambda r: fmt(r.get("frames_multiplas_pessoas"), "")),
        ("tempo no trecho (s)", lambda r: fmt(r.get("t_trecho"), ".4f")),
        ("V média no trecho (m/s)", lambda r: fmt(r.get("v_media"), ".4f")),
        ("média V inst sinal (m/s)", lambda r: fmt(r["resumo_inst"]["media_projetada"], ".4f") if r.get("resumo_inst") else "—"),
        ("média V inst |Δx| (m/s)", lambda r: fmt(r["resumo_inst"]["media_modulo"], ".4f") if r.get("resumo_inst") else "—"),
        ("V média Kalman (m/s)", lambda r: fmt(r.get("kalman", {}).get("v_media_rts"), ".4f")),
        ("média V inst Kalman (m/s)", lambda r: fmt(r.get("kalman", {}).get("media_v_inst_rts"), ".4f")),
        ("dp V Kalman (m/s)", lambda r: fmt(r.get("kalman", {}).get("dp_v_rts_trecho"), ".3f")),
        ("rejeitados Kalman", lambda r: str(r.get("kalman", {}).get("frames_rejeitados", "—"))),
        ("erro V vs cronôm. (%)", lambda r: fmt(r.get("erro_rel"), "+.2f")),
        ("status", lambda r: "válido" if r.get("valido") else "INVÁLIDO"),
    ]

    nomes = [Path(r["arquivo"]).stem for r in resultados]
    largura = max(len(n) for n in nomes) + 2
    print(f"{'':<24}" + "".join(f"{n:>{largura}}" for n in nomes))
    for rotulo, funcao in linhas:
        print(f"{rotulo:<24}" + "".join(f"{funcao(r):>{largura}}" for r in resultados))


def main():
    parser = argparse.ArgumentParser(description="Cruzamentos e velocidade média no trecho a partir dos keypoints.")
    parser.add_argument("keypoints", nargs="+", help="um ou mais CSVs de keypoints")
    parser.add_argument("--calibracao", required=True, help="JSON gerado por src/calibrar.py")
    parser.add_argument("--t-cronometro", type=float, default=None,
                        help="tempo do cronômetro (s) para comparação")
    parser.add_argument("--sem-kalman", action="store_true",
                        help="pula o Kalman (a busca de q e R leva alguns segundos)")
    args = parser.parse_args()

    calibracao = json.loads(resolver(args.calibracao).read_text(encoding="utf-8"))

    resultados = [analisar(resolver(c), calibracao, args.t_cronometro, not args.sem_kalman)
                  for c in args.keypoints]

    if len(resultados) > 1:
        imprimir_tabela(resultados)


if __name__ == "__main__":
    main()
