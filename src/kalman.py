"""
=============================================================================
FILTRO E SUAVIZADOR DE KALMAN PARA A POSIÇÃO DO CENTRO DOS OMBROS
=============================================================================

Estima posição e velocidade a partir das medições de posição z (m), com:

  - modelo de velocidade quase constante: estado x = [p, v], aceleração tratada
    como ruído branco contínuo de intensidade q;
  - passo de tempo variável: quadro sem medição vira só predição (não há
    interpolação nem preenchimento de dado);
  - rejeição de medição pelo teste e²/S > (LIMITE_REJEICAO_DESVIOS)², com os
    quadros rejeitados devolvidos para o relatório;
  - filtro (o que seria possível em tempo real) e suavizador de Rauch, Tung e
    Striebel (offline, sem atraso);
  - q escolhido pelo critério de consistência: média de e²/S igual a 1.

R não é escolhido aqui: é medido nos dados, em analisar_cruzamentos.medir_ruido().

Referência das equações: Aguirre, "Introdução à Identificação de Sistemas", 3. ed.,
2007, cap. 9 (predição, Eq. 9.15 e 9.22; ganho, Eq. 9.30; correção, Eq. 9.16; forma
de Joseph, Eq. 9.28); Särkkä e Svensson, "Bayesian Filtering and Smoothing", 2. ed.,
2023 (matriz Q, Exemplo 4.5; suavizador, Teorema 12.2).

=============================================================================
"""

import math

import numpy as np


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Teste de rejeição: a medição é descartada se e²/S > 5², isto é, se a inovação passa
# de 5 desvios padrão. Em uma curva normal, a chance disso é de cerca de 5,7e-7, então
# o teste quase não corta quadro bom. O objetivo é rejeitar erro de IDENTIDADE (outra
# pessoa ou objeto escolhido pelo YOLO), não ruído nem movimento real.
LIMITE_REJEICAO_DESVIOS = 5.0

# Incerteza inicial da velocidade: valor grande = "não sei a velocidade". O efeito de
# trocá-lo é medido em src/caracterizar_kalman.py.
DP_INICIAL_VELOCIDADE = 10.0     # m/s

# Busca de q: faixa em log10 (m²/s³), pontos da varredura inicial e parada da bisseção.
FAIXA_LOG10_Q = (-3.0, 3.0)
PONTOS_VARREDURA_Q = 60
TOLERANCIA_NIS = 0.005
ITERACOES_MAX_BISSECAO = 60

# Verificação das inovações: maior atraso da autocorrelação, em medições.
ATRASO_MAX = 10

# Matriz de medição: só a posição é medida.
H = np.array([1.0, 0.0])


# =============================================================================
# MODELO
# =============================================================================

def matrizes(dt, q):
    """Matrizes F e Q do modelo de velocidade quase constante para um intervalo dt."""
    F = np.array([[1.0, dt], [0.0, 1.0]])
    Q = q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    return F, Q


# =============================================================================
# FILTRO E SUAVIZADOR
# =============================================================================

def filtrar(t, z, R, q, limite=LIMITE_REJEICAO_DESVIOS, dp_inicial_v=DP_INICIAL_VELOCIDADE):
    """
    Filtro de Kalman sobre todos os instantes t (z = NaN onde não há medição).

    Início: a posição é a primeira medição válida, com variância R, e a velocidade é
    zero, com desvio dp_inicial_v. Essa medição só inicia o estado e não é usada de
    novo. Quadros anteriores a ela ficam no estado inicial, sem predição.

    Medição rejeitada ou ausente: o quadro fica só com a predição.
    """
    n = len(t)
    eye = np.eye(2)

    x_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))
    x_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    F_todos = np.zeros((n, 2, 2))

    e_todos = np.full(n, np.nan)
    S_todos = np.full(n, np.nan)
    rejeitado = np.zeros(n, dtype=bool)

    primeira = int(np.flatnonzero(~np.isnan(z))[0])
    x = np.array([z[primeira], 0.0])
    P = np.diag([R, dp_inicial_v ** 2])

    for k in range(n):
        # Predição (Aguirre, Eq. 9.15 e 9.22).
        if k <= primeira:
            F = eye
        else:
            F, Q = matrizes(t[k] - t[k - 1], q)
            x = F @ x
            P = F @ P @ F.T + Q

        F_todos[k] = F
        x_pred[k] = x
        P_pred[k] = P

        if k > primeira and not np.isnan(z[k]):
            # Inovação e sua variância.
            e = z[k] - H @ x
            PH = P @ H
            S = H @ PH + R

            e_todos[k] = e
            S_todos[k] = S

            if e * e / S > limite ** 2:
                rejeitado[k] = True
            else:
                # Correção: ganho (Eq. 9.30), estado (Eq. 9.16), covariância (Joseph, Eq. 9.28).
                K = PH / S
                x = x + K * e
                IKH = eye - np.outer(K, H)
                P = IKH @ P @ IKH.T + np.outer(K, K) * R

        x_filt[k] = x
        P_filt[k] = P

    return {
        "x_pred": x_pred, "P_pred": P_pred, "x": x_filt, "P": P_filt, "F": F_todos,
        "e": e_todos, "S": S_todos, "rejeitado": rejeitado,
    }


def suavizar_rts(filtro):
    """Suavizador de Rauch, Tung e Striebel: passada para trás sobre o resultado do filtro."""
    x_f, P_f = filtro["x"], filtro["P"]
    x_p, P_p, F = filtro["x_pred"], filtro["P_pred"], filtro["F"]
    n = len(x_f)

    x_s = x_f.copy()
    P_s = P_f.copy()

    for k in range(n - 2, -1, -1):
        G = P_f[k] @ F[k + 1].T @ np.linalg.inv(P_p[k + 1])   # ganho do suavizador
        x_s[k] = x_f[k] + G @ (x_s[k + 1] - x_p[k + 1])
        P_s[k] = P_f[k] + G @ (P_s[k + 1] - P_p[k + 1]) @ G.T

    return x_s, P_s


# =============================================================================
# ESCOLHA DE q
# =============================================================================

def medicoes_aceitas(filtro, no_trecho):
    """Índices das medições aceitas pelo teste de rejeição dentro do trecho."""
    return np.flatnonzero(no_trecho & ~np.isnan(filtro["e"]) & ~filtro["rejeitado"])


def nis_medio(filtro, no_trecho):
    """
    Média de e²/S nas medições aceitas do trecho. Fica fora o início da janela, porque
    ali S ainda depende da incerteza inicial. NaN se nenhuma medição foi aceita.
    """
    idx = medicoes_aceitas(filtro, no_trecho)
    if len(idx) == 0:
        return float("nan")
    return float(np.mean(filtro["e"][idx] ** 2 / filtro["S"][idx]))


def escolher_q(t, z, R, no_trecho, limite=LIMITE_REJEICAO_DESVIOS,
               dp_inicial_v=DP_INICIAL_VELOCIDADE):
    """
    Valor de q em que a média de e²/S vale 1, com R já medido: uma equação com uma incógnita.

    Primeiro, uma varredura em escala logarítmica confere que a média cruza 1 uma única
    vez. Depois, a bisseção divide ao meio o intervalo desse cruzamento. A busca é feita
    em log10(q) para ter a mesma precisão relativa em valores pequenos e grandes.

    Com q muito pequeno, o filtro pode perder o corredor e rejeitar tudo: sem medição
    aceita não há média, e esse caso conta como média acima de 1 (q pequeno demais).

    Sem exatamente um cruzamento na faixa, levanta ValueError. Devolve também a curva da
    varredura, para o relatório.
    """
    def media(log_q):
        return nis_medio(filtrar(t, z, R, 10 ** log_q, limite, dp_inicial_v), no_trecho)

    def acima_de_1(valor):
        return math.isnan(valor) or valor > 1.0

    grade = np.linspace(*FAIXA_LOG10_Q, PONTOS_VARREDURA_Q)
    curva = [media(g) for g in grade]

    trocas = [i for i in range(len(grade) - 1) if acima_de_1(curva[i]) != acima_de_1(curva[i + 1])]
    if len(trocas) != 1 or not acima_de_1(curva[trocas[0]]):
        raise ValueError(
            f"A média de e²/S deveria cruzar 1 uma única vez, descendo, entre q = "
            f"10^{FAIXA_LOG10_Q[0]:g} e 10^{FAIXA_LOG10_Q[1]:g} m²/s³; foram {len(trocas)} cruzamentos."
        )

    a, b = grade[trocas[0]], grade[trocas[0] + 1]
    meio, valor, convergiu = a, curva[trocas[0]], False

    for _ in range(ITERACOES_MAX_BISSECAO):
        meio = (a + b) / 2
        valor = media(meio)
        if not math.isnan(valor) and abs(valor - 1.0) < TOLERANCIA_NIS:
            convergiu = True
            break
        if acima_de_1(valor):
            a = meio
        else:
            b = meio

    return {
        "q": 10 ** meio, "nis_medio": valor, "convergiu": convergiu,
        "curva_q": 10 ** grade, "curva_nis": np.array(curva),
    }


# =============================================================================
# VERIFICAÇÃO DAS INOVAÇÕES
# =============================================================================

def verificar_inovacoes(filtro, no_trecho):
    """
    Evidências sobre as inovações normalizadas e/√S das medições aceitas do trecho. Se o
    modelo explica o que é explicável nos dados:
      - a autocorrelação fica dentro de ±1,96/√N;
      - cerca de 5% dos valores passam de 2 em módulo.
    A média de e²/S não entra: ela vale 1 por construção da escolha de q.
    """
    idx = medicoes_aceitas(filtro, no_trecho)
    e_norm = filtro["e"][idx] / np.sqrt(filtro["S"][idx])
    N = len(e_norm)

    autocorrelacao = [float(np.sum(e_norm[:-tau] * e_norm[tau:]) / np.sum(e_norm ** 2))
                      for tau in range(1, ATRASO_MAX + 1)]
    limite_95 = 1.96 / math.sqrt(N)

    return {
        "N": N,
        "autocorrelacao": autocorrelacao,
        "limite_autocorrelacao": limite_95,
        "atrasos_fora": sum(abs(r) > limite_95 for r in autocorrelacao),
        "fracao_acima_2": float(np.mean(np.abs(e_norm) > 2)),
        "maior_aceita": float(np.max(np.abs(e_norm))),
    }


# =============================================================================
# ESTIMAÇÃO COMPLETA
# =============================================================================

def estimar(t, z, R, no_trecho, q=None, limite=LIMITE_REJEICAO_DESVIOS,
            dp_inicial_v=DP_INICIAL_VELOCIDADE):
    """
    Escolhe q (se não for dado), filtra, suaviza e devolve as séries de velocidade com
    desvio padrão, as evidências das inovações e os quadros rejeitados.
    no_trecho marca os instantes usados na escolha de q e na verificação.
    """
    escolha = None
    if q is None:
        escolha = escolher_q(t, z, R, no_trecho, limite, dp_inicial_v)
        q = escolha["q"]

    filtro = filtrar(t, z, R, q, limite, dp_inicial_v)
    x_s, P_s = suavizar_rts(filtro)

    return {
        "q": q,
        "escolha_q": escolha,
        "nis_medio": nis_medio(filtro, no_trecho),
        "verificacao": verificar_inovacoes(filtro, no_trecho),
        "rejeitado": filtro["rejeitado"],
        "v_filtro": filtro["x"][:, 1], "dp_v_filtro": np.sqrt(filtro["P"][:, 1, 1]),
        "v_rts": x_s[:, 1], "dp_v_rts": np.sqrt(P_s[:, 1, 1]),
        # Posição estimada do ponto medido (ombros) = H·x, para os cruzamentos.
        "z_rts": x_s @ H,
    }
