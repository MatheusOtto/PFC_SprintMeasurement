"""
=============================================================================
CÁLCULO DO SUAVIZADOR NO QUADRO 255 DO ENSAIO C (Apêndice B)
=============================================================================

Só leitura: só imprime, não grava nada. Refaz, passo a passo, o cálculo do suavizador de
Rauch, Tung e Striebel do quadro 255 para o 256 com as funções de kalman.py, com R medido e q
escolhido como no resto do trabalho, e imprime os números do Apêndice B:

  - covariância do filtro no quadro 255 e prevista para o 256;
  - ganho G = P_255 · Fᵀ · (P⁻_256)⁻¹, em três passos (produto, inversa pelo determinante, produto);
  - correção da velocidade no quadro 255 e velocidade no filtro e no suavizador;
  - covariância do suavizador e desvios padrão da velocidade;
  - conferência de que, nesse quadro, as covariâncias coincidem com as do regime permanente,
    calculadas só a partir de q e R.

Uso (na raiz do repositório):  python src/calculo_suavizador.py
=============================================================================
"""

import json
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

ENSAIO = "J_Variando"
QUADRO = 255               # quadro k do filtro; o suavizador usa o quadro k + 1
ITERACOES_REGIME = 2000    # ciclos de predição e correção para chegar ao regime permanente


def covariancia_em_regime(dt, q, R):
    """P (filtro) e P⁻ (prevista) depois de muitos ciclos com dt, q e R fixos."""
    F, Q = kalman.matrizes(dt, q)
    P = np.diag([R, kalman.DP_INICIAL_VELOCIDADE ** 2])
    eye = np.eye(2)
    for _ in range(ITERACOES_REGIME):
        P_prev = F @ P @ F.T + Q
        S = kalman.H @ P_prev @ kalman.H + R
        K = P_prev @ kalman.H / S
        IKH = eye - np.outer(K, kalman.H)
        P = IKH @ P_prev @ IKH.T + np.outer(K, K) * R
    return P, F @ P @ F.T + Q


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    np.set_printoptions(precision=6, linewidth=120)
    calibracao = json.loads((RAIZ / "calibracao" / f"{ENSAIO}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(ac.csv_maior_imgsz(ENSAIO))
    m = ac.medir(dados, calibracao, ac.referencia_cronometro(ENSAIO), com_kalman=True)
    k = m["kalman"]
    R, q = k["R"], k["q"]

    # Mesma série do kalman.estimar(): posição projetada na direção da corrida (m).
    janela = np.isin(dados["frame"], k["frames"])
    z = m["direcao"] * dados["cx"][janela] * m["metros_por_pixel"]
    t = k["t"]
    filtro = kalman.filtrar(t, z, R, q)
    x_s, P_s = kalman.suavizar_rts(filtro)

    i = int(np.flatnonzero(k["frames"] == QUADRO)[0])
    dt = t[i + 1] - t[i]
    F = filtro["F"][i + 1]
    P_f, P_p = filtro["P"][i], filtro["P_pred"][i + 1]

    print(f"{ENSAIO}, quadro {QUADRO}: Δt = {dt:.6f} s (1/60 = {1 / 60:.6f}); R = {R:.4e} m²; q = {q:.4f} m²/s³")
    print(f"\nP_{QUADRO} (filtro):\n{P_f}")
    print(f"P⁻_{QUADRO + 1} (prevista):\n{P_p}")
    print(f"P⁻_vv = P_vv + q·Δt = {P_f[1, 1]:.6f} + {q * dt:.6f} = {P_f[1, 1] + q * dt:.6f}")

    produto = P_f @ F.T
    det = np.linalg.det(P_p)
    inversa = np.linalg.inv(P_p)
    G = produto @ inversa
    print(f"\nP_{QUADRO}·Fᵀ:\n{produto}")
    print(f"det P⁻_{QUADRO + 1} = {det:.4e}")
    print(f"(P⁻_{QUADRO + 1})⁻¹:\n{inversa}")
    print(f"G_{QUADRO}:\n{G}")
    print(f"1 cm a menos de posição reduz a velocidade em {G[1, 0] * 0.01:.3f} m/s")

    # Diferença entre a estimativa suavizada e a prevista pelo filtro no quadro seguinte.
    dif = x_s[i + 1] - filtro["x_pred"][i + 1]
    corr_pos, corr_vel = G[1, 0] * dif[0], G[1, 1] * dif[1]
    print(f"\nquadro {QUADRO + 1}: posição suavizada − prevista = {1000 * dif[0]:.2f} mm; "
          f"velocidade suavizada − prevista = {dif[1]:.3f} m/s")
    print(f"correção da velocidade: {G[1, 0]:.2f}×({dif[0]:.3e}) + {G[1, 1]:.3f}×({dif[1]:.3f}) "
          f"= {corr_pos:.3f} + {corr_vel:.3f} = {corr_pos + corr_vel:.3f} m/s")
    print(f"velocidade no filtro {filtro['x'][i, 1]:.3f} m/s; no suavizador {x_s[i, 1]:.3f} m/s "
          f"(diferença {x_s[i, 1] - filtro['x'][i, 1]:.3f})")

    print(f"\nP^s_vv = {P_s[i, 1, 1]:.7f} m²/s²")
    print(f"desvio padrão da velocidade: filtro √{P_f[1, 1]:.6f} = {np.sqrt(P_f[1, 1]):.3f} m/s; "
          f"suavizador √{P_s[i, 1, 1]:.7f} = {np.sqrt(P_s[i, 1, 1]):.3f} m/s")

    # Regime permanente, só com q e R.
    P_reg, P_reg_p = covariancia_em_regime(dt, q, R)
    print(f"\nregime permanente (só q e R):\nP:\n{P_reg}\nP⁻:\n{P_reg_p}")
    print(f"maior diferença relativa para o quadro {QUADRO}: "
          f"{np.max(np.abs(P_reg / P_f - 1)):.2e} (filtro), {np.max(np.abs(P_reg_p / P_p - 1)):.2e} (prevista)")


if __name__ == "__main__":
    main()
