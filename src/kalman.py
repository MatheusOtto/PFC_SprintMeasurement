"""
=============================================================================
FILTRO E SUAVIZADOR DE KALMAN PARA A POSIÇÃO DO CENTRO DOS OMBROS
=============================================================================

Estima posição e velocidade a partir das medições de posição (m), com:

  - modelo de velocidade quase constante (CV): estado x = [p, v], aceleração
    tratada como ruído branco contínuo de intensidade q;
  - passo de tempo variável: frame sem medição vira só predição (não há
    interpolação nem preenchimento de dado);
  - rejeição de medição pelo teste de inovação (NIS > LIMIAR_INOVACAO), com os
    frames rejeitados devolvidos para o relatório;
  - filtro causal (o que seria possível em tempo real) e suavizador de Kalman
    na forma de Rauch–Tung–Striebel (offline, sem atraso);
  - sintonia de q e de R por máxima verossimilhança das inovações;
  - testes de consistência (NIS médio, fração acima de χ²₁ 95%, brancura).

Referência das equações: Särkkä e Svensson, "Bayesian Filtering and Smoothing",
2. ed., 2023 — Exemplo 4.5 (modelo e matriz Q), Teorema 6.6 (filtro),
Teorema 12.2 (suavizador), Teorema 16.9 (verossimilhança pelas inovações).

Até 2026-10-17 havia também os modelos CA e CV+oscilador, comparados por AIC.
Foram removidos (experimentos/2026-10-07-kalman.md, Revisão C); o código antigo
está no histórico do Git.
=============================================================================
"""

import math

import numpy as np


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Teste de inovação: NIS = ν²/S segue χ² com 1 grau de liberdade se o filtro
# estiver consistente. O objetivo é rejeitar erro de IDENTIDADE (outra pessoa ou
# objeto escolhido pelo YOLO), não ruído nem movimento real.
#
# 25 = 5σ. Escolhido em 2026-10-07 depois de testar 3σ, 4σ, 5σ, 10σ e sem teste
# nos dois vídeos (experimentos/2026-10-07-kalman.md): com 3σ o modelo de
# velocidade quase constante rejeitava 5 frames seguidos de um surto real de
# velocidade dentro da passada (J_Variando 317–321); de 4σ a 10σ o resultado é
# estável (só o frame 279, um falso positivo a 640 px, é rejeitado, e a V muda
# < 0,03%); sem teste, o frame 279 contamina o filtro (NIS médio 21,9).
LIMIAR_INOVACAO = 25.0

# Limiar χ²₁ a 95%, usado só no teste de consistência.
CHI2_1_95 = 3.841

# Incerteza inicial (difusa) da velocidade. Valor grande = "não sei a velocidade".
DP_INICIAL_VELOCIDADE = 10.0     # m/s

# Grade da busca de q por máxima verossimilhança (log10 de q, em m²/s³).
GRADE_Q = np.linspace(-2.0, 4.0, 61)

# R por máxima verossimilhança: fator sobre o desvio σx da segunda diferença
# (R = (fator · σx)²), em busca conjunta com q.
GRADE_FATOR_R = np.linspace(-0.3, 0.8, 23)        # log10: de 0,5× a 6,3×


# =============================================================================
# MODELO
# =============================================================================

class ModeloCV:
    """Velocidade quase constante: x = [p, v], aceleração = ruído branco (q)."""
    nome = "CV"
    dimensao = 2
    H = np.array([1.0, 0.0])

    @staticmethod
    def transicao(dt, p):
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = p["q"] * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        return F, Q

    @staticmethod
    def p0():
        return np.diag([0.0, DP_INICIAL_VELOCIDADE ** 2])

    @staticmethod
    def velocidade(x, P):
        return x[1], P[1, 1]


# =============================================================================
# FILTRO E SUAVIZADOR
# =============================================================================

def filtrar(t, z, r, modelo, params, limiar=LIMIAR_INOVACAO):
    """
    Filtro de Kalman sobre todos os instantes t (z = NaN onde não há medição).

    Início: o estado é a primeira medição da janela (posição com variância R,
    velocidade desconhecida). Essa medição só inicia o estado; não é usada de
    novo na atualização (até 2026-10-17 era: ganho 0,5 no 1º quadro, ver
    experimentos/2026-10-07-kalman.md, Revisão C). Quadros anteriores a ela ficam
    no estado inicial, sem predição.

    Log-verossimilhança: soma de −½(ln 2πS + NIS) das medições usadas, sem as
    primeiras `dimensao` (inicialização difusa: a 1ª fixa a posição, a 2ª a
    velocidade). Medição rejeitada entra com o NIS truncado no limiar, para que
    rejeitar não "melhore" a verossimilhança.
    """
    n, k = len(t), modelo.dimensao
    H = modelo.H

    x_pred = np.zeros((n, k))
    P_pred = np.zeros((n, k, k))
    x_filt = np.zeros((n, k))
    P_filt = np.zeros((n, k, k))
    F_todos = np.zeros((n, k, k))

    inovacao = np.full(n, np.nan)
    variancia_inovacao = np.full(n, np.nan)
    rejeitado = np.zeros(n, dtype=bool)

    primeira = int(np.flatnonzero(~np.isnan(z))[0])
    x = np.zeros(k)
    x[0] = z[primeira]
    P = modelo.p0()
    P[0, 0] = r

    log_ver = 0.0
    usadas = 1            # a primeira medição já está no estado inicial
    inicializacao = [primeira]
    eye = np.eye(k)
    cache = {}   # Δt é quase sempre 1/fps: F e Q calculados uma vez por Δt

    for i in range(n):
        if i <= primeira:
            F = eye
        else:
            dt = round(t[i] - t[i - 1], 9)
            if dt not in cache:
                cache[dt] = modelo.transicao(dt, params)
            F, Q = cache[dt]
            x = F @ x
            P = F @ P @ F.T + Q

        F_todos[i] = F
        x_pred[i] = x
        P_pred[i] = P

        if not np.isnan(z[i]) and i > primeira:
            PH = P @ H
            S = H @ PH + r
            nu = z[i] - H @ x
            nis = nu * nu / S

            inovacao[i] = nu
            variancia_inovacao[i] = S

            if usadas >= k:
                log_ver += -0.5 * (math.log(2 * math.pi * S) + min(nis, limiar))

            if nis <= limiar or usadas < k:
                if usadas < k:
                    inicializacao.append(i)
                K = PH / S
                x = x + K * nu
                IKH = eye - np.outer(K, H)
                P = IKH @ P @ IKH.T + np.outer(K, K) * r   # forma de Joseph
                usadas += 1
            else:
                rejeitado[i] = True

        x_filt[i] = x
        P_filt[i] = P

    return {
        "x_pred": x_pred, "P_pred": P_pred, "x": x_filt, "P": P_filt, "F": F_todos,
        "inovacao": inovacao, "S": variancia_inovacao, "rejeitado": rejeitado,
        "log_verossimilhanca": log_ver, "n_usadas": usadas, "inicializacao": inicializacao,
    }


def suavizar_rts(filtro):
    """Suavizador de Kalman (forma de Rauch–Tung–Striebel): passada para trás sobre o filtro."""
    x_f, P_f = filtro["x"], filtro["P"]
    x_p, P_p, F = filtro["x_pred"], filtro["P_pred"], filtro["F"]
    n = len(x_f)

    x_s = x_f.copy()
    P_s = P_f.copy()

    for i in range(n - 2, -1, -1):
        G = P_f[i] @ F[i + 1].T @ np.linalg.inv(P_p[i + 1])   # ganho do suavizador
        x_s[i] = x_f[i] + G @ (x_s[i + 1] - x_p[i + 1])
        P_s[i] = P_f[i] + G @ (P_s[i + 1] - P_p[i + 1]) @ G.T

    return x_s, P_s


# =============================================================================
# SINTONIA POR MÁXIMA VEROSSIMILHANÇA
# =============================================================================

def _melhor(t, z, r, modelo, candidatos):
    melhor_lv, melhor_p, curva = -np.inf, None, []
    for p in candidatos:
        lv = filtrar(t, z, r, modelo, p)["log_verossimilhanca"]
        curva.append((p, lv))
        if lv > melhor_lv:
            melhor_lv, melhor_p = lv, p
    return melhor_p, melhor_lv, curva


def ajustar(t, z, r, modelo=None):
    """
    q de máxima verossimilhança para um R dado: busca em grade (passos de 10^0,1)
    e refinamento local (passos de 10^0,02). Devolve q, a log-verossimilhança, a
    curva da busca (para o relatório) e se o ótimo caiu na borda da grade.
    """
    modelo = modelo or ModeloCV
    p, lv, curva = _melhor(t, z, r, modelo, [{"q": 10 ** g} for g in GRADE_Q])
    passo = GRADE_Q[1] - GRADE_Q[0]
    g0 = math.log10(p["q"])
    fina = np.linspace(g0 - passo, g0 + passo, 11)
    p, lv, curva_fina = _melhor(t, z, r, modelo, [{"q": 10 ** g} for g in fina])
    curva = sorted(curva + curva_fina, key=lambda c: c[0]["q"])

    return {"params": p, "log_verossimilhanca": lv, "curva": curva,
            "no_limite": parametros_no_limite(p)}


def ajustar_r(t, z, r0):
    """
    Variância de medição por máxima verossimilhança, conjunta com q (grade
    fator × q). Motivo: o σx da segunda diferença é um estimador robusto (MAD) e
    subestima a variância quando o erro do keypoint tem cauda pesada; com R
    pequeno demais, o teste de inovação rejeita medições boas.
    """
    melhor = (-np.inf, None, None)
    for g in GRADE_FATOR_R:
        r = r0 * (10 ** g) ** 2
        for gq in GRADE_Q[::2]:
            lv = filtrar(t, z, r, ModeloCV, {"q": 10 ** gq})["log_verossimilhanca"]
            if lv > melhor[0]:
                melhor = (lv, g, gq)

    _, g, _ = melhor
    passo = GRADE_FATOR_R[1] - GRADE_FATOR_R[0]
    no_limite = g <= GRADE_FATOR_R[0] + passo / 2 or g >= GRADE_FATOR_R[-1] - passo / 2

    return {"r": r0 * (10 ** g) ** 2, "fator": 10 ** g, "no_limite": no_limite}


def parametros_no_limite(p):
    """
    q cujo ótimo caiu na borda da grade grossa: o máximo verdadeiro pode estar
    fora da faixa buscada. Vai para o relatório; não é corrigido em silêncio.
    """
    passo = GRADE_Q[1] - GRADE_Q[0]
    g = math.log10(p["q"])
    return ["q"] if g <= GRADE_Q[0] + passo / 2 or g >= GRADE_Q[-1] - passo / 2 else []


# =============================================================================
# CONSISTÊNCIA
# =============================================================================

def consistencia(filtro):
    """
    NIS das medições aceitas (sem as da inicialização difusa, filtro["inicializacao"]):
      - média ≈ 1 se R e Q estiverem coerentes com os dados;
      - fração acima de χ²₁ 95% ≈ 5%;
      - autocorrelação lag-1 das inovações normalizadas ≈ 0 (brancura),
        calculada só entre frames consecutivos.
    """
    aceitas = ~np.isnan(filtro["inovacao"]) & ~filtro["rejeitado"]
    aceitas[filtro["inicializacao"]] = False
    idx = np.flatnonzero(aceitas)

    nu_norm = filtro["inovacao"][idx] / np.sqrt(filtro["S"][idx])
    nis = nu_norm ** 2

    pares = np.flatnonzero(np.diff(idx) == 1)
    if len(pares) > 2:
        a, b = nu_norm[pares], nu_norm[pares + 1]
        lag1 = float(np.corrcoef(a, b)[0, 1])
    else:
        lag1 = np.nan

    return {
        "nis_medio": float(nis.mean()) if len(nis) else np.nan,
        "fracao_acima_95": float(np.mean(nis > CHI2_1_95)) if len(nis) else np.nan,
        "autocorrelacao_lag1": lag1,
        "n": int(len(nis)),
    }


# =============================================================================
# ESTIMAÇÃO COMPLETA
# =============================================================================

def estimar(t, z, r, modelo=None, params=None):
    """
    Ajusta q (se params for None), filtra, suaviza e devolve as séries de
    velocidade com desvio padrão, mais consistência e frames rejeitados.
    """
    modelo = modelo or ModeloCV
    ajuste = ajustar(t, z, r, modelo) if params is None else {
        "params": params, "log_verossimilhanca": np.nan, "curva": [], "no_limite": []}

    f = filtrar(t, z, r, modelo, ajuste["params"])
    x_s, P_s = suavizar_rts(f)

    def serie(x, P):
        v, var = zip(*(modelo.velocidade(xi, Pi) for xi, Pi in zip(x, P)))
        return np.array(v), np.sqrt(np.maximum(np.array(var), 0.0))

    v_filtro, dp_filtro = serie(f["x"], f["P"])
    v_rts, dp_rts = serie(x_s, P_s)

    return {
        "modelo": modelo.nome,
        "ajuste": ajuste,
        "consistencia": consistencia(f),
        "rejeitado": f["rejeitado"],
        "v_filtro": v_filtro, "dp_v_filtro": dp_filtro,
        "v_rts": v_rts, "dp_v_rts": dp_rts,
        # Posição estimada do ponto medido (ombros) = H·x, para os cruzamentos.
        "z_rts": x_s @ modelo.H,
    }
