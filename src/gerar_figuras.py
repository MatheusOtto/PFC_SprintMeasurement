"""
=============================================================================
FIGURAS DA MONOGRAFIA
=============================================================================

Gera, a partir dos dados brutos (resultados/keypoints) e da calibração, as figuras
de monografia/figuras/. Nenhum cálculo de medição é feito aqui: tudo vem de
analisar_cruzamentos.medir(); este arquivo só desenha.

Figuras:
  velocidade_<ensaio>.pdf   V instantânea bruta (pontos), curva do Kalman suavizado
                            (modelo CV) com faixa de ±2σ, e a velocidade média no
                            trecho pela visão computacional e pelo cronômetro. Usa o
                            CSV de keypoints de maior imgsz (a resolução nativa).
  sintetico_kalman.pdf      teste com dado sintético (src/validar_kalman.py):
                            velocidade verdadeira, bruta e suavizada com faixa ±2σ.

Uso (na pasta do projeto):  python src/gerar_figuras.py
Dependência: matplotlib (vem junto com o ultralytics; não está fixado à parte).
=============================================================================
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

PASTA_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_PROJETO / "src"))

import analisar_cruzamentos as ac  # noqa: E402
import validar_kalman  # noqa: E402

ENSAIOS = ("Trotando_2", "J_Correndo", "J_Variando")
PASTA_SAIDA = PASTA_PROJETO / "monografia" / "figuras"
V_MAX_GRAFICO = 10.0   # m/s; valores fora disso são anotados, não desenhados


def virgula(x, _pos):
    return f"{x:g}".replace(".", ",")


def csv_maior_imgsz(ensaio):
    """CSV de keypoints do ensaio com o maior imgsz (a resolução nativa, padrão do pipeline)."""
    candidatos = []
    for c in (PASTA_PROJETO / "resultados" / "keypoints").glob(f"{ensaio}_imgsz*.csv"):
        sufixo = c.stem.rpartition("_imgsz")[2]
        if sufixo.isdigit():
            candidatos.append((int(sufixo), c))
    return max(candidatos)[1]


def tempo_cronometro(ensaio):
    """Lê o tempo cronometrado de resultados/ensaios.csv (não repetir o número aqui)."""
    caminho = PASTA_PROJETO / "resultados" / "ensaios.csv"
    with open(caminho, encoding="utf-8-sig") as f:
        for linha in csv.DictReader(f):
            if linha["video"] == f"{ensaio}.mp4":
                return float(linha["t_cronometro"])
    raise KeyError(ensaio)


def figura_velocidade(ensaio):
    calibracao = json.loads((PASTA_PROJETO / "calibracao" / f"{ensaio}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(csv_maior_imgsz(ensaio))
    t_cron = tempo_cronometro(ensaio)
    m = ac.medir(dados, calibracao, t_cron, com_kalman=True)

    t0 = m["entrada"]["t"]
    t1 = m["saida"]["t"]
    cv = m["kalman"]

    t_bruta = np.array([a["t"] for a in m["instantaneas"]]) - t0
    v_bruta = np.array([a["v_projetada"] for a in m["instantaneas"]])
    dentro = np.abs(v_bruta) <= V_MAX_GRAFICO
    fora = ~dentro

    tk = cv["t"] - t0
    no_trecho = (tk >= 0) & (tk <= t1 - t0)
    v, dp = cv["v_rts"], cv["dp_v_rts"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    ax.plot(t_bruta[dentro], v_bruta[dentro], ".", color="0.55", markersize=3.5,
            label="Bruta (entre quadros consecutivos)")
    ax.fill_between(tk[no_trecho], (v - 2 * dp)[no_trecho], (v + 2 * dp)[no_trecho],
                    color="tab:blue", alpha=0.18, linewidth=0, label=r"Kalman suavizado $\pm 2\sigma$")
    ax.plot(tk[no_trecho], v[no_trecho], color="tab:blue", linewidth=1.6, label="Kalman suavizado")

    ax.axhline(m["v_media"], color="tab:red", linestyle="--", linewidth=1.0,
               label=f"V média, visão computacional ({m['v_media']:.3f} m/s)".replace(".", ","))
    ax.axhline(m["v_ref"], color="tab:green", linestyle=":", linewidth=1.2,
               label=f"V média, cronômetro ({m['v_ref']:.3f} m/s)".replace(".", ","))

    if fora.any():
        quadros = ", ".join(str(m["instantaneas"][i]["frame"]) for i in np.flatnonzero(fora))
        ax.text(0.99, 0.03, f"{int(fora.sum())} pares fora de escala (quadros {quadros})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")

    # Escala vertical só de exibição: até 1 m/s acima da maior V bruta desenhada.
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(0, min(V_MAX_GRAFICO, np.ceil(v_bruta[dentro].max() + 1.0)))
    ax.set_xlabel("Tempo desde o cruzamento da linha de entrada (s)")
    ax.set_ylabel("Velocidade (m/s)")
    ax.xaxis.set_major_formatter(FuncFormatter(virgula))
    ax.yaxis.set_major_formatter(FuncFormatter(virgula))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    fig.tight_layout()
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDA / f"velocidade_{ensaio}.pdf"
    fig.savefig(destino)
    plt.close(fig)
    print("gerada:", destino.relative_to(PASTA_PROJETO))


def figura_sintetica():
    """Teste com verdade conhecida: os números vêm de validar_kalman.avaliar()."""
    a = validar_kalman.avaliar()
    t, res = a["t"], a["res"]
    v, dp = res["v_rts"], res["dp_v_rts"]
    dentro = np.abs(a["v_bruta"]) <= V_MAX_GRAFICO
    fora = int((~dentro).sum())

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    ax.plot(a["t_bruta"][dentro], a["v_bruta"][dentro], ".", color="0.55", markersize=3.5,
            label="Bruta (entre quadros consecutivos)")
    ax.fill_between(t, v - 2 * dp, v + 2 * dp, color="tab:blue", alpha=0.18, linewidth=0,
                    label=r"Kalman suavizado $\pm 2\sigma$")
    ax.plot(t, v, color="tab:blue", linewidth=1.6, label="Kalman suavizado")
    ax.plot(t, a["v_ombros"], color="black", linewidth=0.8, linestyle="--", label="Verdadeira")

    if fora:
        ax.text(0.99, 0.03, f"{fora} pares fora de escala (erro de identidade)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.35")

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, min(V_MAX_GRAFICO, np.ceil(a["v_bruta"][dentro].max() + 1.0)))
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Velocidade (m/s)")
    ax.xaxis.set_major_formatter(FuncFormatter(virgula))
    ax.yaxis.set_major_formatter(FuncFormatter(virgula))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    fig.tight_layout()
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    destino = PASTA_SAIDA / "sintetico_kalman.pdf"
    fig.savefig(destino)
    plt.close(fig)
    print("gerada:", destino.relative_to(PASTA_PROJETO),
          f"(RMS bruta {a['rms_bruta']:.3f}, suavizada {a['rms']:.3f} m/s)")


def main():
    figura_sintetica()
    for ensaio in ENSAIOS:
        figura_velocidade(ensaio)


if __name__ == "__main__":
    main()
