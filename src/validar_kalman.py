"""
=============================================================================
VALIDAÇÃO DO KALMAN COM DADO SINTÉTICO (verdade conhecida)
=============================================================================

Antes de aplicar o filtro aos vídeos, verifica se ele recupera uma trajetória
cuja velocidade verdadeira é conhecida, com características parecidas com as
medidas no J_Correndo:

  - progressão que acelera e depois desacelera (2,5 → ~5,3 → ~4,5 m/s);
  - oscilação da passada de 1,3 cm a 3,5 Hz (movimento real do ponto medido);
  - ruído branco de medição σ = 0,77 cm (0,58 px × 1,33 cm/px);
  - 10% de frames sem medição + um buraco de 15 frames;
  - um outlier de +8 m (como o falso positivo do frame 279 do J_Variando).

O filtro (modelo CV) estima a velocidade do ponto medido, que inclui a
oscilação da passada; por isso a referência é a velocidade dos ombros.

Critérios (experimentos/2026-10-07-kalman.md, seção 7.1, escritos antes de
rodar esta versão):
  1. o outlier é rejeitado, e nenhuma outra medição;
  2. NIS médio entre 0,7 e 1,3;
  3. a velocidade verdadeira dos ombros fica dentro de ±2σ do suavizador em
     ≥ 90% dos frames;
  4. erro RMS da velocidade dos ombros < 1/3 do erro RMS da velocidade bruta.

Uso:  python src/validar_kalman.py
=============================================================================
"""

import numpy as np

from kalman import estimar


SEMENTE = 2026
FPS = 60.0
DURACAO_S = 7.0
AMPLITUDE_OSC_M = 0.013
FREQ_PASSO_HZ = 3.5
DP_RUIDO_M = 0.0077
FRACAO_FALTANTE = 0.10
FRAME_OUTLIER = 200
OUTLIER_M = 8.0


def gerar():
    rng = np.random.default_rng(SEMENTE)
    t = np.arange(int(DURACAO_S * FPS)) / FPS

    v_prog = 2.5 + 3.0 * (1 - np.exp(-t / 1.2)) - 0.8 * np.clip(t - 4.0, 0, None) / 3.0
    p = np.concatenate([[0.0], np.cumsum((v_prog[1:] + v_prog[:-1]) / 2 / FPS)])

    fase = 2 * np.pi * FREQ_PASSO_HZ * t + 0.3
    s = AMPLITUDE_OSC_M * np.sin(fase)
    v_osc = AMPLITUDE_OSC_M * 2 * np.pi * FREQ_PASSO_HZ * np.cos(fase)

    z = p + s + rng.normal(0, DP_RUIDO_M, len(t))
    z[rng.random(len(t)) < FRACAO_FALTANTE] = np.nan
    z[300:315] = np.nan
    z[FRAME_OUTLIER] = p[FRAME_OUTLIER] + OUTLIER_M

    return t, z, v_prog, v_prog + v_osc


def avaliar():
    """
    Roda o filtro no dado sintético e devolve as séries e as métricas. Usada por
    main() e por src/gerar_figuras.py (figura do teste), para os dois mostrarem os
    mesmos números.
    """
    t, z, v_prog, v_ombros = gerar()
    r = DP_RUIDO_M ** 2

    validos = ~np.isnan(z)
    i = np.flatnonzero(validos)
    consecutivos = np.diff(i) == 1
    t_bruta = t[i[1:]][consecutivos]
    v_bruta = (np.diff(z[i]) * FPS)[consecutivos]
    v_bruta_ref = v_ombros[i[1:]][consecutivos]
    ok = np.abs(v_bruta) < 50     # tira só os dois pares do outlier, para referência
    rms_bruta = np.sqrt(np.mean((v_bruta[ok] - v_bruta_ref[ok]) ** 2))

    res = estimar(t, z, r)

    return {
        "t": t, "z": z, "v_prog": v_prog, "v_ombros": v_ombros,
        "t_bruta": t_bruta, "v_bruta": v_bruta, "rms_bruta": rms_bruta,
        "res": res,
        "rms": np.sqrt(np.mean((res["v_rts"] - v_ombros) ** 2)),
        "rms_prog": np.sqrt(np.mean((res["v_rts"] - v_prog) ** 2)),
        "cobertura": np.mean(np.abs(res["v_rts"] - v_ombros) <= 2 * res["dp_v_rts"]),
        "rejeitados": np.flatnonzero(res["rejeitado"]).tolist(),
        "nis": res["consistencia"]["nis_medio"],
        "validos": int(validos.sum()),
    }


def main():
    a = avaliar()
    t, res = a["t"], a["res"]
    rms_bruta, rms, rms_prog = a["rms_bruta"], a["rms"], a["rms_prog"]
    cobertura, rejeitados, nis = a["cobertura"], a["rejeitados"], a["nis"]

    print(f"Sintético: {len(t)} frames, {a['validos']} medições, outlier no frame {FRAME_OUTLIER}")
    print(f"V inst bruta (Δz/Δt): RMS do erro vs ombros = {rms_bruta:.3f} m/s")
    print(f"Kalman CV: q = {res['ajuste']['params']['q']:.3g} m²/s³ | NIS médio {nis:.2f} | "
          f"rejeitados {rejeitados}")
    print(f"           RMS vs ombros = {rms:.3f} m/s | RMS vs progressão = {rms_prog:.3f} m/s | "
          f"cobertura ±2σ (ombros) = {100 * cobertura:.1f}%")

    criterios = [
        (f"1. outlier rejeitado e nenhuma outra medição (rejeitados: {rejeitados})",
         rejeitados == [FRAME_OUTLIER]),
        (f"2. NIS médio entre 0,7 e 1,3 (foi {nis:.2f})", 0.7 <= nis <= 1.3),
        (f"3. cobertura ±2σ dos ombros ≥ 90% (foi {100 * cobertura:.1f}%)", cobertura >= 0.90),
        (f"4. RMS < 1/3 do RMS da bruta (foi {rms:.3f} contra {rms_bruta / 3:.3f} m/s)",
         rms < rms_bruta / 3),
    ]

    print()
    for texto, passou in criterios:
        print(f"[{'PASSOU' if passou else 'FALHOU'}] {texto}")

    print(f"\n{'VALIDAÇÃO APROVADA' if all(p for _, p in criterios) else 'VALIDAÇÃO REPROVADA'}")


if __name__ == "__main__":
    main()
