"""
=============================================================================
FIGURAS DOS RESULTADOS
=============================================================================

Gera em saidas/figuras/ as figuras dos três ensaios. Nenhum cálculo de medição é feito
aqui: tudo vem de analisar_cruzamentos.medir() e de kalman.py; este arquivo só desenha.

  figura_4.1a e 4.1b   Ensaio A: deslocamento e segunda diferença, por cor e por pose
  figura_4.2a e 4.2b   Ensaio B: posição medida e posição filtrada
  figura_4.2c          Ensaio B: previsão, medição e estimativa em cinco quadros
  nis_q                média de e²/S em função de q, nos três ensaios
  figura_4.3a          Ensaio C: posição medida, filtrada e suavizada
  figura_4.3           Ensaio C: velocidade bruta, do filtro e do suavizador

Uso (na raiz do repositório):  python src/gerar_figuras.py
=============================================================================
"""

import functools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

PASTA_PROJETO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_PROJETO / "src"))

import analisar_cruzamentos as ac  # noqa: E402
import kalman  # noqa: E402
import segmentacao_hsv  # noqa: E402

ENSAIOS = ("Trotando_2", "J_Correndo", "J_Variando")
PASTA_SAIDA = PASTA_PROJETO / "saidas" / "figuras"
V_MAX_GRAFICO = 10.0   # m/s; valores fora disso não são desenhados

# Largura de 15 cm, fonte de 10 pt e cores com estilos de linha diferentes.
LARGURA_POL = 15.0 / 2.54
COR_PRIORI, COR_MEDIDA, COR_POSTERIORI = "#1f5fa8", "#e07b24", "#2e8b57"
COR_REJEITADO = "#b22222"
COR_MEDIANA = "#333333"
COR_ENSAIO = {"Trotando_2": COR_PRIORI, "J_Correndo": COR_MEDIDA, "J_Variando": COR_POSTERIORI}
NOME_ENSAIO = {"Trotando_2": "Ensaio A", "J_Correndo": "Ensaio B", "J_Variando": "Ensaio C"}
ESTILO_METODO = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
}
QUADROS_3D = range(380, 385)           # cinco quadros seguidos no meio do trecho do Ensaio B
LARGURA_SINO_DESVIOS = 3.5             # cada sino da figura 3D é desenhado até ±3,5 desvios


def virgula(x, _pos):
    return f"{x:g}".replace(".", ",")


def numero(valor, casas):
    """Número com vírgula decimal e sinal de menos tipográfico."""
    return f"{valor:.{casas}f}".replace(".", ",").replace("-", "−")


def eixo_decimal(casas=None):
    """Formatador de eixo com vírgula; sem `casas`, usa o menor número de dígitos."""
    if casas is None:
        return FuncFormatter(lambda v, _p: f"{v:g}".replace(".", ",").replace("-", "−"))
    return FuncFormatter(lambda v, _p: numero(v, casas))


@functools.lru_cache(maxsize=None)
def medicao(ensaio):
    """Mede o ensaio com o pipeline; a escolha de q leva alguns segundos e só roda uma vez."""
    calibracao = json.loads((PASTA_PROJETO / "calibracao" / f"{ensaio}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(ac.csv_maior_imgsz(ensaio))
    return dados, ac.medir(dados, calibracao, ac.referencia_cronometro(ensaio), com_kalman=True)


def filtro_do_ensaio(ensaio):
    """
    Passagem do filtro com a janela, o R e o q devolvidos por medir(). kalman.estimar() não
    devolve as inovações nem a priori, então o filtro roda de novo com as mesmas entradas, e a
    igualdade dos quadros rejeitados confere que é a mesma passagem.
    """
    dados, m = medicao(ensaio)
    cv = m["kalman"]
    z = m["direcao"] * dados["cx"][np.isin(dados["frame"], cv["frames"])] * m["metros_por_pixel"]
    filtro = kalman.filtrar(cv["t"], z, cv["R"], cv["q"])
    if not np.array_equal(filtro["rejeitado"], cv["rejeitado"]):
        raise ValueError(f"{ensaio}: a passagem do filtro difere da de medir().")
    return m, z, filtro


def salvar(fig, nome):
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_SAIDA / f"{nome}.png", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"gerada: {(PASTA_SAIDA / nome).relative_to(PASTA_PROJETO)}.png")


def figura_nis_q():
    """Média de e²/S na varredura devolvida por kalman.escolher_q(), com o q escolhido marcado."""
    estilos = {"Trotando_2": ("-", "o", (-6, -18), "right"),
               "J_Correndo": ((0, (5, 2)), "s", (8, 12), "left"),
               "J_Variando": ((0, (1, 1.2)), "^", (-2, 20), "right")}
    with plt.rc_context(ESTILO_METODO):
        fig, ax = plt.subplots(figsize=(LARGURA_POL, 3.35), layout="constrained")
        for ensaio, (linha, marcador, deslocamento, alinhamento) in estilos.items():
            _, m = medicao(ensaio)
            escolha = m["kalman"]["escolha_q"]
            q, nis = escolha["curva_q"], escolha["curva_nis"]
            sem_media = np.isnan(nis)
            rotulo = NOME_ENSAIO[ensaio]
            if sem_media.any():
                ultimo = q[np.flatnonzero(sem_media)[-1]]
                rotulo += f" (nenhuma medição aceita até $q$ = {numero(ultimo, 3)})"
                print(f"  {NOME_ENSAIO[ensaio]}: sem medição aceita em {int(sem_media.sum())} dos {len(q)} "
                      f"pontos, até q = {ultimo:.4f} m²/s³")
            cor = COR_ENSAIO[ensaio]
            ax.plot(q, nis, color=cor, linestyle=linha, linewidth=1.3, marker=marcador, markersize=7,
                    markevery=[], markerfacecolor="white", markeredgecolor=cor, label=rotulo)
            ax.plot(escolha["q"], escolha["nis_medio"], linestyle="none", marker=marcador, markersize=7,
                    markerfacecolor="white", markeredgecolor=cor, markeredgewidth=1.2, zorder=5)
            ax.annotate(f"$q$ = {numero(escolha['q'], 3)}", xy=(escolha["q"], escolha["nis_medio"]),
                        xytext=deslocamento, textcoords="offset points", ha=alinhamento)

            print(f"  {NOME_ENSAIO[ensaio]}: q = {escolha['q']:.4f} m²/s³, média de e²/S = {escolha['nis_medio']:.4f}")

        ax.axhline(1.0, color="0.45", linewidth=1.0, zorder=0, label="Média igual a 1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1e3)
        ax.set_xlabel(r"Intensidade do ruído de processo, $q$ (m$^2$/s$^3$)")
        ax.set_ylabel(r"NIS médio, média de $e_k^2/S_k$")
        ax.yaxis.set_major_formatter(eixo_decimal())
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda _v, _p: ""))
        ax.grid(which="major", color="0.88", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="lower left")
        salvar(fig, "nis_q")


def figura_priori_medida_posteriori_3d(ensaio="J_Correndo"):
    """
    Priori, medida e posteriori em quadros seguidos. O eixo x é a posição em relação a um ponto
    que anda à velocidade média desses quadros: sem isso, os sinos ficariam espalhados nos
    cerca de 40 cm que o atleta percorre em 5 quadros.
    """
    m, z, filtro = filtro_do_ensaio(ensaio)
    cv = m["kalman"]
    quadros = list(QUADROS_3D)
    indices = {k: int(np.flatnonzero(cv["frames"] == k)[0]) for k in quadros}
    t = {k: cv["t"][i] for k, i in indices.items()}

    sigma_z = np.sqrt(cv["R"]) * 100
    sinos = {
        k: {
            "priori": (filtro["x_pred"][i][0] * 100, np.sqrt(filtro["P_pred"][i][0, 0]) * 100),
            "medida": (z[i] * 100, sigma_z),
            "posteriori": (filtro["x"][i][0] * 100, np.sqrt(filtro["P"][i][0, 0]) * 100),
        }
        for k, i in indices.items()
    }

    primeiro, ultimo = quadros[0], quadros[-1]
    p_inicial = sinos[primeiro]["priori"][0]
    v_ref = (sinos[ultimo]["priori"][0] - p_inicial) / (t[ultimo] - t[primeiro])     # cm/s

    def relativa(k, media):
        return media - (p_inicial + v_ref * (t[k] - t[primeiro]))

    def densidade(x, media, dp):
        return np.exp(-0.5 * ((x - media) / dp) ** 2) / (dp * np.sqrt(2 * np.pi))

    estilos = (("priori", (0, (4, 2)), 1.3, COR_PRIORI), ("medida", "-", 1.3, COR_MEDIDA),
               ("posteriori", "-", 2.0, COR_POSTERIORI))
    extensao = max(abs(relativa(k, s[nome][0])) + LARGURA_SINO_DESVIOS * s[nome][1]
                   for k, s in sinos.items() for nome, *_ in estilos)
    limite_x = float(np.ceil(extensao))
    pico = max(densidade(s[n][0], s[n][0], s[n][1]) for s in sinos.values() for n, *_ in estilos)

    with plt.rc_context(ESTILO_METODO):
        fig = plt.figure(figsize=(LARGURA_POL, 4.3))
        ax = fig.add_axes([0.0, 0.02, 1.0, 1.04], projection="3d")

        for k in reversed(quadros):                                       # de trás para a frente
            for nome, estilo, espessura, cor in estilos:
                media, dp = sinos[k][nome]
                mr = relativa(k, media)
                x = np.linspace(mr - LARGURA_SINO_DESVIOS * dp, mr + LARGURA_SINO_DESVIOS * dp, 200)
                y = densidade(x, mr, dp)
                if nome == "posteriori":
                    ax.add_collection3d(Poly3DCollection([list(zip(x, [k] * len(x), y))], facecolor=cor,
                                                         alpha=0.25, edgecolor="none"))
                ax.plot(x, [k] * len(x), y, color=cor, linewidth=espessura, linestyle=estilo)

        # Previsão: da posteriori de um quadro à priori do seguinte, no chão do gráfico.
        for k0, k1 in zip(quadros[:-1], quadros[1:]):
            a, b = relativa(k0, sinos[k0]["posteriori"][0]), relativa(k1, sinos[k1]["priori"][0])
            ax.plot([a, b], [k0, k1], [0, 0], color="0.3", linewidth=0.9, linestyle=(0, (1, 1.2)))
            ax.scatter([a], [k0], [0], color=COR_POSTERIORI, s=9, depthshade=False)
            ax.scatter([b], [k1], [0], color=COR_PRIORI, s=9, depthshade=False)

        dt_ms = (t[ultimo] - t[primeiro]) / (ultimo - primeiro) * 1000
        ax.set_xlim(-limite_x, limite_x)
        ax.set_ylim(primeiro - 0.3, ultimo + 0.3)
        ax.set_zlim(0, np.ceil(pico * 10) / 10)
        ax.set_xlabel(f"Posição relativa a um ponto\na {numero(v_ref / 100, 2)} m/s (cm)", labelpad=9)
        ax.set_ylabel(f"Quadro ({numero(dt_ms, 1)} ms cada)", labelpad=10)
        ax.set_zlabel("Densidade de\nprobabilidade (1/cm)", labelpad=8)
        ax.set_yticks(quadros)
        ax.set_xticks(np.arange(-np.floor(limite_x / 4) * 4, limite_x + 0.01, 4))
        ax.tick_params(pad=1)
        ax.xaxis.set_major_formatter(eixo_decimal())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0f}"))
        ax.zaxis.set_major_formatter(eixo_decimal())
        ax.view_init(elev=27, azim=-50)
        ax.set_box_aspect((1.0, 2.7, 0.7), zoom=1.08)

        fig.legend(
            handles=[
                Line2D([], [], color=COR_PRIORI, linewidth=1.3, linestyle=(0, (4, 2)), label="Priori (previsão do modelo)"),
                Line2D([], [], color=COR_MEDIDA, linewidth=1.3, label="Medida (YOLO)"),
                Line2D([], [], color=COR_POSTERIORI, linewidth=2.0, label="Posteriori (estimativa)"),
                Line2D([], [], color="0.3", linewidth=0.9, linestyle=(0, (1, 1.2)), label="Posteriori → priori do quadro seguinte"),
            ],
            loc="upper center", ncol=2, frameon=False, columnspacing=1.5, labelspacing=0.3,
        )
        salvar(fig, "figura_4.2c_ensaio_b_ciclos")

    print(f"  {NOME_ENSAIO[ensaio]}: q = {cv['q']:.3f} m²/s³; sigma da medida {sigma_z:.3f} cm; "
          f"v_ref {v_ref / 100:.3f} m/s")
    for k in quadros:
        i = indices[k]
        print(f"  quadro {k}: desvio da priori {sinos[k]['priori'][1]:.3f} cm; da posteriori {sinos[k]['posteriori'][1]:.3f} cm; "
              f"K_p {filtro['P_pred'][i][0, 0] / filtro['S'][i]:.3f}; "
              f"inovação {filtro['e'][i] / np.sqrt(filtro['S'][i]):+.2f} desvios")


def serie_medida(dados, m):
    """Deslocamento e segunda diferença da posição medida, orientados no sentido da corrida."""
    quadros = dados["frame"]
    janela = (quadros >= m["entrada"]["frame_antes"]) & (quadros <= m["saida"]["frame_depois"])
    q, x = quadros[janela], dados["cx"][janela]
    fora = ~np.isin(m["d"], m["d_usados"])
    return q, m["direcao"] * np.diff(x), m["direcao"] * m["d"], fora


def limites_da_serie(passo, d, topo_d=None):
    """Escalas verticais pelos valores normais, sem o erro grosseiro."""
    normal_passo = np.abs(passo - np.median(passo)) < 100
    normal_d = np.abs(d - np.median(d)) < 100
    lim_passo = (passo[normal_passo].min() - 3, passo[normal_passo].max() + 3)
    lim_d = (d[normal_d].min() - 3, topo_d if topo_d is not None else d[normal_d].max() * 1.7)
    return lim_passo, lim_d


def figura_deslocamento(q, passo, d, nome, cor, limites, rotulo_d, fora=None, cor_mediana=COR_MEDIANA, rejeitado=None):
    """Deslocamento entre quadros (acima) e segunda diferença (abaixo). fora marca os d descartados."""
    marca = dict(linestyle="none", marker="X", markersize=8, markerfacecolor="white",
                 markeredgecolor=COR_REJEITADO, markeredgewidth=1.2, zorder=5)
    q_passo, q_d = q[1:], q[1:-1]
    with plt.rc_context(ESTILO_METODO):
        fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(LARGURA_POL, 4.3), layout="constrained")
        a.plot(q_passo, passo, color=cor, linewidth=1.0, marker="o", markersize=3)
        a.axhline(np.median(passo), color=cor_mediana, linewidth=1.1, linestyle=(0, (5, 2)),
                  label=f"Deslocamento mediano: {numero(np.median(passo), 1)} px")
        a.set_ylabel("Deslocamento entre\nquadros (pixels)")
        a.legend(frameon=False, loc="upper left")

        if fora is None:
            b.plot(q_d, d, color=cor, linewidth=0, marker="o", markersize=3, label=f"{rotulo_d} ({len(d)})")
        else:
            mediana, corte = np.median(d), ac.CORTE_GROSSEIRO_DESVIOS * ac.desvio_mad(d)
            visivel = (d >= limites[1][0]) & (d <= limites[1][1])
            b.plot(q_d[~fora], d[~fora], color=cor, linewidth=0, marker="o", markersize=3,
                   label=f"{rotulo_d} usados ({int((~fora).sum())})")
            b.plot(q_d[fora & visivel], d[fora & visivel], label=f"{rotulo_d} descartados ({int(fora.sum())})", **marca)
            for sinal in (-1, 1):
                b.axhline(mediana + sinal * corte, color=COR_REJEITADO, linewidth=1.0, linestyle=(0, (5, 2)),
                          label=f"Limite de exclusão: ±{numero(corte, 2)} px" if sinal == 1 else None)
        if rejeitado is not None:
            b.plot(q_d[q_d == rejeitado], d[q_d == rejeitado], label=f"Quadro {rejeitado} rejeitado", **marca)
        b.set_xlabel("Quadro")
        b.set_ylabel("Segunda diferença $d$\n(pixels)")
        b.legend(frameon=False, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.8,
                 bbox_to_anchor=(0.5, 1.04))

        a.set_ylim(limites[0])
        b.set_ylim(limites[1])
        for ax in (a, b):
            ax.yaxis.set_major_formatter(eixo_decimal())
            ax.spines[["top", "right"]].set_visible(False)
        salvar(fig, nome)


def figuras_ensaio_a():
    """Ensaio A: segmentação por cor e estimação de pose, nas mesmas escalas verticais."""
    dados_hsv, m_hsv = segmentacao_hsv.medir_hsv()
    q, passo, d, fora = serie_medida(dados_hsv, m_hsv)
    limites = limites_da_serie(passo, d)
    figura_deslocamento(q, passo, d, "figura_4.1a_hsv_deslocamento", COR_PRIORI, limites, "$d$", fora, COR_MEDIDA)

    dados, m = medicao("Trotando_2")
    q, passo, d, fora = serie_medida(dados, m)
    figura_deslocamento(q, passo, d, "figura_4.1b_pose_deslocamento", COR_PRIORI, limites, "$d$", fora, COR_MEDIDA)


def figuras_filtro(ensaio, prefixo, topo_d=None, rejeitado=None):
    """Posição medida, filtrada e suavizada de um ensaio, nas mesmas escalas verticais."""
    dados, m = medicao(ensaio)
    _, _, filtro = filtro_do_ensaio(ensaio)
    cv, C = m["kalman"], m["metros_por_pixel"]

    q, passo, d, fora = serie_medida(dados, m)
    limites = limites_da_serie(passo, d, topo_d)
    figura_deslocamento(q, passo, d, prefixo["medida"], COR_PRIORI, limites, "$d$", fora)

    selecao = np.isin(cv["frames"], q)
    series = {"filtrada": (filtro["x"][selecao, 0] / C, COR_MEDIDA, "$d$ da posição do filtro"),
              "suavizada": (cv["z_rts"][selecao] / C, COR_POSTERIORI, "$d$ da posição suavizada")}
    for chave, (posicao, cor, rotulo) in series.items():
        if chave in prefixo:
            passo_e = np.diff(posicao)
            figura_deslocamento(q, passo_e, np.diff(passo_e), prefixo[chave], cor, limites, rotulo, rejeitado=rejeitado)


def figura_velocidades_c(ensaio="J_Variando", rejeitado=279):
    """Ensaio C: velocidade bruta, do filtro e do suavizador, com a faixa de ±2 desvios."""
    m, _, filtro = filtro_do_ensaio(ensaio)
    cv = m["kalman"]
    fr, t = cv["frames"], cv["t"]
    x_s, P_s = kalman.suavizar_rts(filtro)
    vf, vs, dp = filtro["x"][:, 1], x_s[:, 1], np.sqrt(P_s[:, 1, 1])
    tr = (t >= cv["t_entrada_rts"]) & (t <= cv["t_saida_rts"])

    q_b = np.array([a["frame"] for a in m["instantaneas"]])
    v_b = np.array([a["v_projetada"] for a in m["instantaneas"]])
    dentro = np.abs(v_b) <= V_MAX_GRAFICO
    print(f"  bruta fora da escala: {np.round(v_b[~dentro], 2).tolist()} m/s")

    def extremos(serie):
        k_max, k_min = fr[tr][np.argmax(serie[tr])], fr[tr][np.argmin(serie[tr])]
        print(f"  máx {serie[tr].max():.3f} no quadro {k_max}; mín {serie[tr].min():.3f} no quadro {k_min}")
        return (k_max, k_min), (serie[tr].max(), serie[tr].min())

    q_f, v_f = extremos(vf)
    q_s, v_s = extremos(vs)
    v_rej = vs[np.flatnonzero(fr == rejeitado)[0]]

    with plt.rc_context(ESTILO_METODO):
        fig, ax = plt.subplots(figsize=(LARGURA_POL, 3.6), layout="constrained")
        ax.plot(q_b[dentro], v_b[dentro], ".", color="0.6", markersize=3, label="Bruta (entre quadros consecutivos)")
        ax.fill_between(fr[tr], (vs - 2 * dp)[tr], (vs + 2 * dp)[tr], color=COR_POSTERIORI, alpha=0.22, linewidth=0,
                        label="Suavizador: faixa de ±2 desvios")
        ax.plot(fr[tr], vf[tr], color=COR_MEDIDA, linewidth=0.9, linestyle=(0, (4, 2)), label="Filtro")
        ax.plot(fr[tr], vs[tr], color=COR_POSTERIORI, linewidth=1.5, label="Suavizador")
        ax.plot(q_f, v_f, linestyle="none", marker="^", markersize=7, markerfacecolor=COR_MEDIDA,
                markeredgecolor="black", markeredgewidth=0.7, zorder=6,
                label=f"Extremos do filtro (quadros {q_f[0]} e {q_f[1]})")
        ax.plot(q_s, v_s, linestyle="none", marker="o", markersize=6, markerfacecolor=COR_POSTERIORI,
                markeredgecolor="black", markeredgewidth=0.7, zorder=6,
                label=f"Extremos do suavizador (quadros {q_s[0]} e {q_s[1]})")
        ax.plot([rejeitado], [v_rej], linestyle="none", marker="X", markersize=8, markerfacecolor="white",
                markeredgecolor=COR_REJEITADO, markeredgewidth=1.2, zorder=6,
                label=f"Quadro {rejeitado} (medição rejeitada)")

        ax.set_xlim(fr[tr][0], fr[tr][-1])
        ax.set_ylim(0, V_MAX_GRAFICO)
        ax.set_xlabel("Quadro")
        ax.set_ylabel("Velocidade (m/s)")
        ax.yaxis.set_major_formatter(eixo_decimal())
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=1.0, handlelength=1.8, fontsize=8.5,
                  labelspacing=0.3)
        salvar(fig, "figura_4.3_ensaio_c_velocidades")


def main():
    figuras_ensaio_a()
    figuras_filtro("J_Correndo", {"medida": "figura_4.2a_ensaio_b_medida",
                                  "filtrada": "figura_4.2b_ensaio_b_filtrada"})
    figura_priori_medida_posteriori_3d()
    figura_nis_q()
    figuras_filtro("J_Variando", {"medida": "figura_4.3a_ensaio_c_medida",
                                  "filtrada": "figura_4.3a_ensaio_c__filtrada",
                                  "suavizada": "figura_4.3a_ensaio_c__suavizada"},
                   topo_d=10.5, rejeitado=279)
    figura_velocidades_c()


if __name__ == "__main__":
    main()
