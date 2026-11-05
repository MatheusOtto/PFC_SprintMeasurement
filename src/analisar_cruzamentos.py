"""
=============================================================================
CRUZAMENTOS E VELOCIDADE MÉDIA
(CSV de keypoints + calibração → tempos, velocidade, diagnóstico)
=============================================================================

Lê o CSV gerado por src/extrair_keypoints.py e calcula, sem rodar o YOLO:

  - ponto medido: centro dos ombros (COCO 5 e 6), os dois com conf ≥ 0,35;
  - frames sem medição válida não são preenchidos: o intervalo simplesmente cresce;
  - entrada: medição anterior fora da zona e atual dentro dela;
  - cruzamento interpolado linearmente entre as duas medições válidas que cercam
    a linha, com 0 ≤ alfa ≤ 1;
  - velocidade média no trecho = distância / (t_saída − t_entrada).

Além disso:
  - CRITÉRIO DE GAP MÁXIMO no cruzamento (GAP_MAX_CRUZAMENTO_S), que só sinaliza:
    o valor continua no relatório, marcado como inválido;
  - base de tempo pelo carimbo de cada quadro, e não por quadro / FPS do contêiner
    (ver carregar_keypoints). O tempo nominal fica no relatório
    como referência;
  - velocidade instantânea em duas versões lado a lado: com sinal, projetada na
    direção da passagem, e com |Δx|;
  - comparação com o cronômetro, sem e com filtro, numa função única
    (comparar_com_cronometro), com a incerteza da própria referência;
  - vários CSVs de uma vez, com tabela lado a lado no final.

Duas velocidades instantâneas ficam lado a lado: a BRUTA, par a par, sem nenhum filtro, e a
do filtro e do suavizador de Kalman (src/kalman.py). A velocidade média também é dada sem
filtro (interpolação dos cruzamentos) e com filtro (cruzamentos da posição suavizada).
O visualizador (src/visualizar_medicao.py) usa a função medir() deste arquivo,
então os números da tela e os do relatório vêm do mesmo cálculo.

Uso (na raiz do repositório):

    python src/analisar_cruzamentos.py resultados/keypoints/J_Correndo_imgsz1920.csv \\
        --calibracao calibracao/J_Correndo.json

O tempo do cronômetro vem de resultados/cronometragens.csv, se --t-cronometro não for dado.
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

# Os dois ombros precisam satisfazer esse limiar.
CONFIANCA_MIN_OMBRO = 0.35

# -----------------------------------------------------------------------------
# GAP MÁXIMO PARA INTERPOLAR UM CRUZAMENTO
#
# Critério aplicado igualmente a todos os ensaios. Cruzamento com gap acima do limite NÃO
# é descartado: o valor continua no relatório, marcado como inválido.
#
# Justificativa (orçamento de erro): a interpolação linear entre duas medições
# separadas por Δt erra a posição em até a·Δt²/8 se o corredor acelera com a.
# Em tempo, δt ≈ a·Δt² / (8·v). Com a = 5 m/s² (conservador, saída de sprint) e
# v = 5 m/s, Δt = 0,25 s dá δt ≈ 8 ms por cruzamento: menos de meio período de
# quadro a 30 fps e ~0,2% de um trecho de 3,5 s. 0,25 s é também ~1 passada a
# 4 Hz. Acima disso o erro cresce com Δt² e, como no J_Correndo com imgsz 640
# (gap de 3,15 s), o par de medições pode nem pertencer ao mesmo movimento.
# -----------------------------------------------------------------------------
GAP_MAX_CRUZAMENTO_S = 0.25

# Kalman (src/kalman.py): janela além dos cruzamentos, para o suavizador RTS ter
# medições dos dois lados de cada linha.
MARGEM_KALMAN_S = 0.5

# Medição de R: uma segunda diferença é erro grosseiro (por exemplo, outra pessoa
# detectada) se ficar a mais deste número de desvios preliminares da mediana. Regra fixa,
# sem exclusão manual. Não confundir com kalman.LIMITE_REJEICAO_DESVIOS, que é o teste do
# filtro. Efeito de trocar o valor: src/caracterizar_kalman.py (4, 5 e 10).
CORTE_GROSSEIRO_DESVIOS = 5.0

# Referência: os 10 tempos de cada ensaio ficam em resultados/cronometragens.csv, cronometrados
# assistindo ao vídeo. A referência é a média de todos eles, sem retirar nenhum.
ARQUIVO_CRONOMETRAGENS = PASTA_PROJETO / "resultados" / "cronometragens.csv"

# Erro absoluto médio da cronometragem manual (Hetzler et al., 2008), convertido em desvio
# padrão u_H (normal) da incerteza do operador. O valor vem de cronometragem ao vivo, e
# não foi medido para tempos tomados assistindo ao vídeo.
ERRO_CRONOMETRO_LITERATURA_S = 0.15


# =============================================================================
# LEITURA
# =============================================================================

def resolver(caminho):
    caminho = Path(caminho)
    return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho


def csv_maior_imgsz(video):
    """CSV de keypoints do vídeo com o maior imgsz (a resolução nativa, padrão do pipeline)."""
    candidatos = []
    for c in (PASTA_PROJETO / "resultados" / "keypoints").glob(f"{video}_imgsz*.csv"):
        sufixo = c.stem.rpartition("_imgsz")[2]
        if sufixo.isdigit():
            candidatos.append((int(sufixo), c))
    return max(candidatos)[1]


def referencia_de_tempos(tempos):
    """
    Referência a partir das cronometragens: média t_medio e desvio padrão s. Com um único
    tempo (opção --t-cronometro) não há como medir a dispersão, e s fica 0.
    """
    tempos = np.array(tempos, dtype=float)
    return {
        "t_medio": float(tempos.mean()),
        "s": float(tempos.std(ddof=1)) if len(tempos) > 1 else 0.0,
        "n": len(tempos),
        "tempos": tempos.tolist(),
    }


def referencia_cronometro(video):
    """
    Referência de um vídeo (nome sem extensão), com todos os tempos de
    resultados/cronometragens.csv. Devolve None se não houver tempos. Como foram medidos
    assistindo ao vídeo, a comparação não detecta erro no relógio do próprio vídeo.
    """
    with open(ARQUIVO_CRONOMETRAGENS, encoding="utf-8-sig") as arquivo:
        tempos = [float(l["t_cronometro"]) for l in csv.DictReader(arquivo) if l["video"] == video]

    return referencia_de_tempos(tempos) if tempos else None


def carregar_keypoints(caminho):
    """
    Lê o CSV por nome de coluna. Aceita também arquivos só com o tempo nominal
    (timestamp_s) e sem n_pessoas.
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

    # Base de tempo: o carimbo de cada quadro (CAP_PROP_POS_MSEC). O tempo nominal divide o índice do quadro pela taxa média do
    # contêiner, que nos vídeos a 60 fps vem distorcida pelo primeiro intervalo
    # anômalo (59,979 e 60,077 fps; os carimbos dão 60). Ele fica só como referência
    # no relatório e como reserva para arquivos sem carimbo.
    tem_carimbo = dados["t_carimbo"] is not None
    dados["t"] = dados["t_carimbo"] if tem_carimbo else dados["t_nominal"]
    dados["base_tempo"] = "carimbo" if tem_carimbo else "nominal"

    xe = coluna_kp(INDICE_OMBRO_ESQUERDO, "x")
    ye = coluna_kp(INDICE_OMBRO_ESQUERDO, "y")
    ce = coluna_kp(INDICE_OMBRO_ESQUERDO, "conf")
    xd = coluna_kp(INDICE_OMBRO_DIREITO, "x")
    yd = coluna_kp(INDICE_OMBRO_DIREITO, "y")
    cd = coluna_kp(INDICE_OMBRO_DIREITO, "conf")

    # Critério de validade: coordenadas positivas e os dois
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
    """Interpolação linear, na base oficial e, como referência, na nominal."""
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
    da passagem: primeiro a entrada (define a direção), depois a saída.
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

def segundas_diferencas(serie, frames):
    """
    Segunda diferença d[k] = x[k+1] − 2x[k] + x[k−1], em cada trio de quadros consecutivos
    com medição válida. Trio com quadro sem medição fica de fora, sem preenchimento.
    """
    d = []

    for k in range(1, len(frames) - 1):
        consecutivos = frames[k] - frames[k - 1] == 1 and frames[k + 1] - frames[k] == 1
        if consecutivos and not np.isnan(serie[k - 1:k + 2]).any():
            d.append(serie[k + 1] - 2 * serie[k] + serie[k - 1])

    return np.array(d)


def desvio_mad(d):
    """Desvio pela mediana dos desvios absolutos; 1,4826 = 1/0,6745 converte para a curva normal."""
    return 1.4826 * np.median(np.abs(d - np.median(d)))


def medir_ruido(d, corte=CORTE_GROSSEIRO_DESVIOS):
    """
    Desvio σx do ruído de uma medição, em pixels, a partir das segundas diferenças d.
    Um único valor por vídeo: cada d usa 3 quadros, e σx junta todos os d do trecho.

      1. desvio preliminar de d pela mediana dos desvios absolutos (MAD), pouco afetado
         por erros grosseiros;
      2. descarta os d a mais de `corte` desvios preliminares da mediana;
      3. σx = desvio padrão dos d restantes / √6.

    Depois, R = (σx · C)², em metros².

    O √6 não é R nem incerteza de R. É o fator de conversão entre o desvio padrão da segunda
    diferença d e o desvio do tremor de uma medição: d = x[k+1] − 2x[k] + x[k−1] tem variância
    (1 + 4 + 1)·σ² = 6·σ² quando o ruído é independente entre quadros. O movimento quase não
    aparece em 3 quadros.

    O MAD não é a estimativa final porque R é a variância, e o MAD só estima a variância para
    ruído gaussiano. Por isso, o MAD serve só para o corte dos erros grosseiros.
    """
    if len(d) < 5:
        return {"sigma_x_px": np.nan, "sigma_mad_px": np.nan, "razao_desvio_mad": np.nan,
                "curtose_d": np.nan, "n_trincas": len(d), "d_descartados": 0, "d_usados": d}

    desvio_preliminar = desvio_mad(d)
    usados = d[np.abs(d - np.median(d)) <= corte * desvio_preliminar]

    sigma_x = usados.std(ddof=1) / np.sqrt(6)
    sigma_mad = desvio_preliminar / np.sqrt(6)
    padronizados = (usados - usados.mean()) / usados.std(ddof=1)

    return {
        "sigma_x_px": sigma_x,
        "sigma_mad_px": sigma_mad,
        "razao_desvio_mad": sigma_x / sigma_mad,
        "curtose_d": float(np.mean(padronizados ** 4)),   # 3 na curva normal
        "n_trincas": len(d),
        "d_descartados": len(d) - len(usados),
        "d_usados": usados,
    }


def diagnosticar_trecho(dados, entrada, saida):
    """
    Trecho = do frame que confirmou a entrada ao frame que confirmou a saída,
    inclusive.
    """
    frames = dados["frame"]
    no_trecho = (frames >= entrada["frame_depois"]) & (frames <= saida["frame_depois"])
    validos = no_trecho & ~np.isnan(dados["cx"])

    t_validos = dados["t"][validos]
    maior_gap_s = float(np.max(np.diff(t_validos))) if validos.sum() > 1 else np.nan

    janela = (frames >= entrada["frame_antes"]) & (frames <= saida["frame_depois"])
    d = segundas_diferencas(np.where(janela, dados["cx"], np.nan), frames)

    if dados["n_pessoas"] is not None:
        multiplas = int(np.sum(no_trecho & (dados["n_pessoas"] > 1)))
    else:
        multiplas = None

    return {
        "frames_trecho": int(no_trecho.sum()),
        "validos_trecho": int(validos.sum()),
        "disponibilidade": 100.0 * validos.sum() / max(no_trecho.sum(), 1),
        "maior_gap_s": maior_gap_s,
        "d": d,
        **medir_ruido(d),
        "frames_multiplas_pessoas": multiplas,
    }


# =============================================================================
# VELOCIDADE INSTANTÂNEA
# =============================================================================

def calcular_velocidades_instantaneas(dados, entrada, saida, direcao, metros_por_pixel):
    """
    Uma velocidade por par de medições válidas consecutivas dentro do trecho,
    do frame que confirmou a entrada até a última
    medição antes da linha de saída (o par que cruza a saída não entra). Frames
    sem medição não são preenchidos: o Δt do par simplesmente cresce.

    Duas versões, lado a lado:
      v_projetada = direção · Δx · escala / Δt   (com sinal; negativa = suspeita)
      v_modulo    = |Δx| · escala / Δt           (com módulo)
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


def aplicar_kalman(dados, m, R=None, q=None, limite=kalman.LIMITE_REJEICAO_DESVIOS,
                   dp_inicial_v=kalman.DP_INICIAL_VELOCIDADE, margem=MARGEM_KALMAN_S):
    """
    Roda o filtro e o suavizador de Kalman (src/kalman.py) sobre a posição do centro dos
    ombros projetada na direção da corrida (m), numa janela em volta do trecho. O trecho
    usado na escolha de q vai dos cruzamentos da interpolação linear, que não dependem de q.

    Devolve um único dicionário: R medido, q escolhido, evidências das inovações, quadros
    rejeitados, séries de velocidade (filtro e suavizador, com desvio padrão) e a velocidade
    média no trecho pelo Kalman: os instantes em que a posição SUAVIZADA dos ombros cruza as
    duas linhas. R e os demais argumentos só mudam nos testes de sensibilidade.
    """
    frames = dados["frame"]
    t = dados["t"]
    entrada, saida = m["entrada"], m["saida"]
    C = m["metros_por_pixel"]

    janela = (t >= entrada["t"] - margem) & (t <= saida["t"] + margem)
    t_j = t[janela]
    f_j = frames[janela]
    z_j = m["direcao"] * dados["cx"][janela] * C
    no_trecho = (t_j >= entrada["t"]) & (t_j <= saida["t"])

    if R is None:
        if not np.isfinite(m["sigma_x_px"]):
            raise ValueError("Sem trios de quadros consecutivos no trecho para medir R.")
        R = (m["sigma_x_px"] * C) ** 2

    z_entrada = m["direcao"] * entrada["x_linha"] * C
    z_saida = m["direcao"] * saida["x_linha"] * C

    res = kalman.estimar(t_j, z_j, R, no_trecho, q, limite, dp_inicial_v)

    t_ent = cruzamento_continuo(t_j, res["z_rts"], z_entrada)
    t_sai = cruzamento_continuo(t_j, res["z_rts"], z_saida, t_ent if t_ent else -np.inf)

    res["frames"] = f_j
    res["t"] = t_j
    res["frames_rejeitados"] = f_j[res["rejeitado"]].tolist()
    res["R"] = R

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
        # ombros, que inclui a oscilação da passada. Tempo contado desde a entrada.
        idx = np.flatnonzero(no_trecho)
        for nome, j in (("min", idx[np.argmin(res["v_rts"][idx])]),
                        ("max", idx[np.argmax(res["v_rts"][idx])])):
            res[f"v_{nome}_rts"] = float(res["v_rts"][j])
            res[f"t_v_{nome}_rts"] = float(t_j[j] - t_ent)
            res[f"dp_v_{nome}_rts"] = float(res["dp_v_rts"][j])
            res[f"frame_v_{nome}_rts"] = int(f_j[j])

    return res


# =============================================================================
# COMPARAÇÃO COM O CRONÔMETRO
# =============================================================================

def incerteza_nao_comum(dados, calibracao, m):
    """
    U_s: incerteza expandida relativa (k = 2) do sistema, só com o que não é comum com a
    referência: o ruído de posição nos dois cruzamentos e a resolução dos cliques da calibração.
    A distância D não entra, porque se cancela na diferença para o cronômetro.

      u_t de cada linha = σx / v_local (v_local em pixels por segundo, entre os quadros que cercam a linha)
      u_ruído = √(u_t,entrada² + u_t,saída²) / t_trecho
      u_clique = a·√2/√3 / distância_px     cada clique é retangular de ±a; a diferença de dois
      U_s = 2·√(u_ruído² + u_clique²)
    """
    f = dados["frame"]
    u_t = []
    for lado in ("entrada", "saida"):
        c = m[lado]
        x0 = float(dados["cx"][f == c["frame_antes"]][0])
        x1 = float(dados["cx"][f == c["frame_depois"]][0])
        dt = float(dados["t"][f == c["frame_depois"]][0] - dados["t"][f == c["frame_antes"]][0])
        u_t.append(m["sigma_x_px"] / (abs(x1 - x0) / dt))
    u_ruido = np.hypot(*u_t) / m["t_trecho"]

    a_clique = calibracao.get("resolucao_clique_px", 1.0)
    u_clique = a_clique * np.sqrt(2) / np.sqrt(3) / abs(m["x_max"] - m["x_min"])
    return float(2.0 * np.hypot(u_ruido, u_clique))


def comparar_com_cronometro(v_sistema, referencia, distancia, U_s):
    """
    Compara uma velocidade média do sistema com o cronômetro. Função única: a tela e os
    relatórios mostram os mesmos números.

      v_ref = D / t_medio
      ERRO BRUTO:  erro_bruto (%) = 100·(v_sistema − v_ref) / v_ref
      INCERTEZA DA REFERÊNCIA:
        u_A = s / √n                       parte aleatória, Tipo A
        u_H = 0,15·√(π/2) = 0,188 s        operador: erro absoluto médio de 0,15 s (Hetzler et al.,
                                           2008) convertido em desvio padrão de uma normal
        U_t = 2·√(u_A² + u_H²)             em quadratura, k = 2
        U_ref (%) = 100·U_t / t_medio
      COMPATIBILIDADE:
        E_n = |erro_bruto| / √(U_s² + U_ref²), compatível se E_n ≤ 1
        U_s (%) é a incerteza do sistema sem o que é comum com a referência (incerteza_nao_comum).

    Como ler:
      - o erro bruto supõe que o cronômetro é exato;
      - o erro do sistema não pode ser isolado com precisão melhor que a da referência.
    """
    t_medio, s, n = referencia["t_medio"], referencia["s"], referencia["n"]
    v_ref = distancia / t_medio

    erro_bruto_ms = v_sistema - v_ref
    erro_bruto = 100.0 * erro_bruto_ms / v_ref

    u_A = s / np.sqrt(n)
    # fonte: Hetzler et al., 2008 (erro absoluto médio 0,15 s); MAE de normal = σ·√(2/π)
    u_H = ERRO_CRONOMETRO_LITERATURA_S * np.sqrt(np.pi / 2)
    U_t = 2.0 * np.hypot(u_A, u_H)
    U_ref_v = 100.0 * U_t / t_medio
    U_s_v = 100.0 * U_s

    E_n = abs(erro_bruto) / np.hypot(U_s_v, U_ref_v)
    compativel = bool(E_n <= 1.0)

    return {
        "v_sistema": v_sistema,
        "v_ref": v_ref,
        "t_medio": t_medio,
        "s": s,
        "n": n,
        "erro_bruto": erro_bruto,
        "erro_bruto_ms": erro_bruto_ms,
        "u_A": u_A,
        "u_H": u_H,
        "U_t": U_t,
        "U_ref_v": U_ref_v,
        "U_s_v": U_s_v,
        "E_n": E_n,
        "compativel": compativel,
        "veredito": ("compatível com a referência (E_n ≤ 1)" if compativel
                     else "não compatível com a referência (E_n > 1)"),
    }


# =============================================================================
# MEDIÇÃO (sem impressão; usada pelo relatório e pelo visualizador)
# =============================================================================

def medir(dados, calibracao, referencia=None, com_kalman=True):
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

    if referencia is not None:
        U_s = incerteza_nao_comum(dados, calibracao, m)
        sem_filtro = comparar_com_cronometro(m["v_media"], referencia, distancia, U_s)
        m["comparacao"] = {"sem_filtro": sem_filtro}
        m["t_cronometro"] = referencia["t_medio"]
        m["v_ref"] = sem_filtro["v_ref"]
        m["erro_v"] = sem_filtro["erro_bruto_ms"]
        m["erro_rel"] = sem_filtro["erro_bruto"]
        m["erro_t"] = m["t_trecho"] - referencia["t_medio"]

    m["valido"] = entrada["gap_ok"] and saida["gap_ok"]

    if com_kalman:
        m["kalman"] = aplicar_kalman(dados, m)
        if referencia is not None and "v_media_rts" in m["kalman"]:
            m["comparacao"]["com_filtro"] = comparar_com_cronometro(
                m["kalman"]["v_media_rts"], referencia, distancia, U_s)

    return m


# =============================================================================
# RELATÓRIO
# =============================================================================

def imprimir_comparacao(m):
    """Comparação com o cronômetro em formato didático, sem e com filtro."""
    comparacao = m["comparacao"]
    c = comparacao["sem_filtro"]

    print(f"\nCOMPARAÇÃO COM O CRONÔMETRO (tempos medidos assistindo ao vídeo)")
    print(f"  V do cronômetro: {c['v_ref']:.4f} m/s | t médio {c['t_medio']:.4f} s | "
          f"s = {c['s']:.4f} s ({c['n']} tempos)")
    print(f"  Incerteza da referência U_ref: {c['U_ref_v']:.2f}% ({c['U_t']:.4f} s), com "
          f"u_A = {c['u_A']:.4f} s (aleatória) e u_H = {c['u_H']:.3f} s (operador, Hetzler et al., 2008)")
    print(f"  Incerteza do sistema U_s: {c['U_s_v']:.2f}%")

    for rotulo, chave in (("Sem filtro", "sem_filtro"), ("Com filtro", "com_filtro")):
        if chave not in comparacao:
            continue
        c = comparacao[chave]
        print(f"  {rotulo}:")
        print(f"    V do sistema: {c['v_sistema']:.4f} m/s")
        print(f"    Erro em relação ao cronômetro (bruto): {c['erro_bruto']:+.2f}% "
              f"({c['erro_bruto_ms']:+.4f} m/s)")
        print(f"    E_n: {c['E_n']:.2f}")
        print(f"    Veredito: {c['veredito']}")

    print("  O erro bruto supõe cronômetro exato.")


def analisar(caminho, calibracao, referencia, com_kalman=True):
    dados = carregar_keypoints(caminho)
    meta = dados["meta"]
    m = medir(dados, calibracao, referencia, com_kalman)

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
    print(f"σx centro dos ombros: {m['sigma_x_px']:.3f} px "
          f"(≈ {m['sigma_x_px'] * m['metros_por_pixel'] * 100:.2f} cm, "
          f"{m['d_descartados']} de {m['n_trincas']} trincas descartadas) | "
          f"pelo MAD {m['sigma_mad_px']:.3f} px | razão {m['razao_desvio_mad']:.3f} | "
          f"curtose {m['curtose_d']:.2f}"
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
              f"média |Δx| {r['media_modulo']:.4f} m/s | "
              f"viés do abs {r['media_modulo'] - r['media_projetada']:+.4f} m/s | "
              f"pares negativos {r['negativas']}")
        print(f"                dp {r['desvio_projetada']:.3f} m/s | "
              f"min {r['minimo_projetada']:.3f} | max {r['maximo_projetada']:.3f} | "
              f"média com sinal − V média: {r['media_projetada'] - m['v_media']:+.4f} m/s "
              f"({100.0 * (r['media_projetada'] - m['v_media']) / m['v_media']:+.2f}%)")

    if "kalman" in m:
        imprimir_kalman(m)

    if "comparacao" in m:
        imprimir_comparacao(m)

    if m["valido"]:
        print(f"STATUS: VÁLIDO (gaps ≤ {GAP_MAX_CRUZAMENTO_S * 1000:.0f} ms)")
    else:
        print(f"STATUS: INVÁLIDO — gap de cruzamento acima de "
              f"{GAP_MAX_CRUZAMENTO_S * 1000:.0f} ms. Valor reportado, não usar como medição.")

    return resultado


def imprimir_kalman(m):
    k = m["kalman"]
    print(f"\nKALMAN — janela ±{MARGEM_KALMAN_S} s em volta do trecho | "
          f"R = ({m['sigma_x_px']:.3f} px)² = {k['R']:.3e} m² | "
          f"rejeição acima de {kalman.LIMITE_REJEICAO_DESVIOS:g} desvios")
    print(f"  {'q (m²/s³)':>10}{'NIS':>7}{'>2σ':>7}{'autocorr.':>11}{'limite':>8}{'fora/10':>9}  "
          f"{'V média':>9}{'média V':>9}{'dp V':>7}{'±σ médio':>9}  rejeitados")

    c = k["verificacao"]
    v_media = f"{k['v_media_rts']:.4f}" if "v_media_rts" in k else "—"
    media = f"{k['media_v_inst_rts']:.4f}" if "media_v_inst_rts" in k else "—"
    dp = f"{k['dp_v_rts_trecho']:.3f}" if "dp_v_rts_trecho" in k else "—"
    sig = f"{k['dp_medio_v_rts']:.3f}" if "dp_medio_v_rts" in k else "—"
    print(f"  {k['q']:>10.4g}{k['nis_medio']:>7.3f}{100 * c['fracao_acima_2']:>6.1f}%"
          f"{c['autocorrelacao'][0]:>+11.3f}{c['limite_autocorrelacao']:>8.3f}{c['atrasos_fora']:>9}  "
          f"{v_media:>9}{media:>9}{dp:>7}{sig:>9}  {k['frames_rejeitados']}")

    if k["escolha_q"] is not None and not k["escolha_q"]["convergiu"]:
        print("  AVISO: a bisseção de q terminou sem atingir a tolerância da média de e²/S.")

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
        ("σx ombros (px)", lambda r: fmt(r.get("sigma_x_px"), ".3f")),
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
                        help="tempo do cronômetro (s); sem ele, usa resultados/cronometragens.csv")
    parser.add_argument("--sem-kalman", action="store_true",
                        help="pula o Kalman (a busca de q leva alguns segundos)")
    args = parser.parse_args()

    calibracao = json.loads(resolver(args.calibracao).read_text(encoding="utf-8"))

    if args.t_cronometro is not None:
        referencia = referencia_de_tempos([args.t_cronometro])
    else:
        referencia = referencia_cronometro(Path(calibracao["video"]).stem)

    resultados = [analisar(resolver(c), calibracao, referencia, not args.sem_kalman)
                  for c in args.keypoints]

    if len(resultados) > 1:
        imprimir_tabela(resultados)


if __name__ == "__main__":
    main()
