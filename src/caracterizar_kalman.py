"""
=============================================================================
CARACTERIZAÇÃO DO FILTRO DE KALMAN (modelo CV) NOS ENSAIOS
=============================================================================

Somente leitura: só imprime, não grava nada. Resultados e leitura em
diagnostico/kalman_caracterizacao.md.

Para cada ensaio (CSV de keypoints de maior imgsz), com o q e o R escolhidos por
máxima verossimilhança no próprio ensaio, como no relatório do analisador:

  1. parâmetros e o que significam: q, R, fator κ sobre σx, variação típica da
     velocidade de um quadro para o outro (√(q·Δt));
  2. ganho de Kalman K = P⁻·Hᵀ/S nas primeiras medições e em regime;
  3. teste de inovação: inovação dos quadros rejeitados e maior inovação aceita, em
     desvios padrão (√NIS), e a chance de uma medição boa passar de 3, 4, 5 e 10σ se
     o erro fosse gaussiano;
  4. desvio padrão médio da velocidade no trecho: filtro causal × suavizador;
  5. consistência: NIS médio, fração acima de χ²₁ 95% e autocorrelação lag-1 das
     inovações normalizadas, com a tolerância ±2/√n;
  6. resposta em frequência na velocidade: senoide de posição de 1 cm somada a uma
     progressão constante, sem ruído; ganho = amplitude estimada / amplitude
     verdadeira, atraso pela fase. Filtro causal e suavizador.

Uso (na pasta do projeto):  python src/caracterizar_kalman.py
=============================================================================
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import analisar_cruzamentos as ac  # noqa: E402
import kalman  # noqa: E402


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Ensaio -> tempo do cronômetro (s), como em src/orcamento_erro.py.
ENSAIOS = {"Trotando_2": 2.55, "J_Correndo": 3.48, "J_Variando": 7.23}

# Ganho "em regime": mediana a partir desta medição (o ganho converge em ~10).
INICIO_REGIME = 20

# Resposta em frequência: duração do sinal sintético e margem descartada nas pontas.
DURACAO_SENOIDE_S = 12.0
MARGEM_SENOIDE_S = 3.0
AMPLITUDE_SENOIDE_M = 0.01
VELOCIDADE_PROGRESSAO = 3.0
FREQUENCIAS_HZ = (0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 15.0, 20.0)


def csv_maior_imgsz(nome):
    candidatos = []
    for c in (RAIZ / "resultados" / "keypoints").glob(f"{nome}_imgsz*.csv"):
        sufixo = c.stem.rpartition("_imgsz")[2]
        if sufixo.isdigit():
            candidatos.append((int(sufixo), c))
    return max(candidatos)[1]


def resposta(fps, q, r, f):
    """Ganho e atraso (ms) na velocidade, filtro causal e suavizador, para uma senoide em f."""
    t = np.arange(int(DURACAO_SENOIDE_S * fps)) / fps
    w = 2 * math.pi * f
    z = VELOCIDADE_PROGRESSAO * t + AMPLITUDE_SENOIDE_M * np.sin(w * t)
    filtro = kalman.filtrar(t, z, r, kalman.ModeloCV, {"q": q}, limiar=1e12)
    x_s, _ = kalman.suavizar_rts(filtro)
    sel = (t > MARGEM_SENOIDE_S) & (t < DURACAO_SENOIDE_S - MARGEM_SENOIDE_S)
    c, s = np.cos(w * t[sel]), np.sin(w * t[sel])

    saida = []
    for v in (filtro["x"][:, 1], x_s[:, 1]):
        y = v[sel] - VELOCIDADE_PROGRESSAO
        a, b = 2 * np.mean(y * c), 2 * np.mean(y * s)     # v ≈ a·cos + b·sin
        ganho = math.hypot(a, b) / (AMPLITUDE_SENOIDE_M * w)
        atraso_ms = math.atan2(b, a) / w * 1000            # verdadeira: cos(wt)
        saida += [ganho, atraso_ms]
    return saida


def caracterizar(nome, t_crono):
    csv_kp = csv_maior_imgsz(nome)
    cal = json.loads((RAIZ / "calibracao" / f"{nome}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(csv_kp)
    m = ac.medir(dados, cal, t_crono, com_kalman=True)
    k = m["kalman"]
    cv = k
    q = float(cv["ajuste"]["params"]["q"])
    r = float(k["r"])
    mpp = m["metros_por_pixel"]

    t = dados["t"]
    entrada, saida = m["entrada"], m["saida"]
    janela = ((t >= entrada["t"] - ac.MARGEM_KALMAN_S)
              & (t <= saida["t"] + ac.MARGEM_KALMAN_S))
    tj = t[janela]
    fj = dados["frame"][janela]
    zj = m["direcao"] * dados["cx"][janela] * mpp
    dt = float(np.median(np.diff(tj)))
    fps = 1.0 / dt

    f = kalman.filtrar(tj, zj, r, kalman.ModeloCV, {"q": q})
    H = kalman.ModeloCV.H

    print("\n" + "=" * 78)
    print(f"{nome} ({csv_kp.name})   {fps:.2f} fps   escala {mpp * 100:.4f} cm/px")
    print("=" * 78)

    # ------------------------------------------------------------ 1. parâmetros
    print("\n[1] Parâmetros (máxima verossimilhança no próprio ensaio)")
    print(f"   q = {q:.4f} m²/s³  ->  variação típica da velocidade em um quadro "
          f"√(q·Δt) = {math.sqrt(q * dt):.3f} m/s; em 1 s √q = {math.sqrt(q):.2f} m/s")
    print(f"   R = {r:.3e} m²  ->  √R = {math.sqrt(r) * 100:.3f} cm = {math.sqrt(r) / mpp:.3f} px")
    print(f"   σx (segunda diferença, MAD) = {k['sigma_x_px']:.3f} px;  fator κ = √R/(σx·C) = "
          f"{k['fator_r']:.4f}")

    # ------------------------------------------------------------ 2. ganho
    # Medições na ordem; a 1ª só inicia o estado (sem ganho), as demais atualizam.
    medidas = np.flatnonzero(~np.isnan(zj))
    K = np.array([f["P_pred"][i] @ H / (H @ f["P_pred"][i] @ H + r) for i in medidas])
    print("\n[2] Ganho de Kalman K = P⁻·Hᵀ/S  [posição (adim.); velocidade (1/s)]")
    print("    1ª medição: inicia o estado (posição = medição, velocidade desconhecida); sem ganho")
    for n in (2, 3, 5, 10):
        if n <= len(K):
            print(f"   {n:>2}ª medição: [{K[n - 1, 0]:.3f}; {K[n - 1, 1]:6.2f}]")
    k_pos = float(np.median(K[INICIO_REGIME - 1:, 0]))
    k_vel = float(np.median(K[INICIO_REGIME - 1:, 1]))
    print(f"   em regime (mediana a partir da {INICIO_REGIME}ª): [{k_pos:.3f}; {k_vel:.2f}]"
          f"   (α = {k_pos:.3f}, β = K_vel·Δt = {k_vel * dt:.3f})")
    print(f"   leitura: a posição estimada anda {100 * k_pos:.1f}% em direção a cada medida; "
          f"1 cm de inovação muda a velocidade em {k_vel:.1f} cm/s")

    # ------------------------------------------------------------ 3. teste de inovação
    nu, S, rej = f["inovacao"], f["S"], f["rejeitado"]
    sig = np.abs(nu) / np.sqrt(S)
    aceitas = ~np.isnan(nu) & ~rej
    print(f"\n[3] Teste de inovação (limiar NIS > {kalman.LIMIAR_INOVACAO:g}, i.e. "
          f"{math.sqrt(kalman.LIMIAR_INOVACAO):g}σ)")
    print(f"   medições na janela: {len(medidas)} (a 1ª inicia o estado); √S mediano = "
          f"{np.median(np.sqrt(S[aceitas])) * 100:.3f} cm")
    for i in np.flatnonzero(rej):
        print(f"   REJEITADO quadro {fj[i]}: inovação {nu[i]:+.3f} m, √S = {math.sqrt(S[i]) * 100:.3f} cm "
              f"-> {sig[i]:.0f}σ (NIS {sig[i] ** 2:.3g})")
    if not rej.any():
        print("   nenhum quadro rejeitado")
    i_max = np.flatnonzero(aceitas)[np.argmax(sig[aceitas])]
    print(f"   maior inovação aceita: {sig[i_max]:.2f}σ no quadro {fj[i_max]} "
          f"(inovação {nu[i_max] * 100:+.2f} cm)")
    print(f"   aceitas acima de 3σ: {int(np.sum(sig[aceitas] > 3))}; acima de 4σ: "
          f"{int(np.sum(sig[aceitas] > 4))}")
    n_usadas = int(aceitas.sum())
    print(f"   se o erro fosse gaussiano, rejeições esperadas de medições boas em {n_usadas}:")
    for n_sig in (3, 4, 5, 10):
        p = math.erfc(n_sig / math.sqrt(2))
        print(f"      {n_sig:>2}σ: P = {p:.3e}  ->  {n_usadas * p:.4f}")

    # ------------------------------------------------------------ 4. filtro × suavizador
    tt = cv["t"]
    no_trecho = (tt >= cv["t_entrada_rts"]) & (tt <= cv["t_saida_rts"])
    print("\n[4] Desvio padrão médio da velocidade no trecho")
    print(f"   filtro causal {np.mean(cv['dp_v_filtro'][no_trecho]):.3f} m/s  |  suavizador "
          f"{np.mean(cv['dp_v_rts'][no_trecho]):.3f} m/s")

    # ------------------------------------------------------------ 5. consistência
    c = cv["consistencia"]
    print("\n[5] Consistência (medições aceitas, sem as 2 da inicialização)")
    print(f"   NIS médio {c['nis_medio']:.3f} (esperado 1) | acima de 3,84: "
          f"{100 * c['fracao_acima_95']:.1f}% (esperado 5%) | autocorrelação lag-1 "
          f"{c['autocorrelacao_lag1']:+.3f} (tolerância ±{2 / math.sqrt(c['n']):.3f}, n = {c['n']})")

    # ------------------------------------------------------------ 6. resposta em frequência
    print("\n[6] Resposta em frequência na velocidade (ganho; atraso em ms)")
    print(f"   {'f (Hz)':>7} | {'filtro':>7} {'atraso':>7} | {'suaviz.':>7} {'atraso':>7}")
    for fr in FREQUENCIAS_HZ:
        if fr > 0.9 * fps / 2:      # na frequência de Nyquist a senoide some na amostragem
            continue
        g_f, a_f, g_s, a_s = resposta(fps, q, r, fr)
        print(f"   {fr:>7.1f} | {g_f:>7.3f} {a_f:>7.1f} | {g_s:>7.3f} {a_s:>7.1f}")
    for rotulo, indice in (("filtro", 0), ("suavizador", 2)):
        fs = np.arange(0.5, 0.9 * fps / 2, 0.1)
        ganhos = [resposta(fps, q, r, fr)[indice] for fr in fs]
        abaixo = [fr for fr, g in zip(fs, ganhos) if g < 1 / math.sqrt(2)]
        print(f"   frequência de corte (ganho 0,707) do {rotulo}: "
              f"{abaixo[0]:.1f} Hz" if abaixo else f"   {rotulo}: ganho > 0,707 até {fs[-1]:.1f} Hz")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for nome, t_crono in ENSAIOS.items():
        caracterizar(nome, t_crono)


if __name__ == "__main__":
    main()
