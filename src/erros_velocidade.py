"""
=============================================================================
ERROS DE MEDIÇÃO DA VELOCIDADE NOS ENSAIOS
=============================================================================

Somente leitura: só imprime, não grava nada. Reúne, para cada ensaio:

  1. a comparação com o cronômetro, sem e com filtro, pela função única
     analisar_cruzamentos.comparar_com_cronometro(): erro bruto, incertezas da referência
     (U_ref) e do sistema (U_s), E_n e veredito;
  2. a contribuição do processamento para a incerteza da velocidade média: ruído de posição nos
     cruzamentos e calibração. É a precisão do cálculo, não o erro do sistema;
  3. a velocidade instantânea: incerteza prevista e dispersão. Nos vídeos não há referência
     quadro a quadro, então aqui não existe erro medido.

Como os tempos foram medidos assistindo ao vídeo, a comparação não detecta erro no relógio do
próprio vídeo.

Uso (na raiz do repositório):  python src/erros_velocidade.py
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


ENSAIOS = {"A": "Trotando_2", "B": "J_Correndo", "C": "J_Variando"}

JANELA_TENDENCIA_S = 0.5
LIMITE_PAR_GROSSEIRO = 50.0    # m/s: tira da dispersão os pares do erro de identidade


# =============================================================================
# CONTRIBUIÇÃO DO PROCESSAMENTO, EM FRAÇÃO DA VELOCIDADE MÉDIA
# =============================================================================

def erro_ruido_posicao(dados, m):
    """
    O ruído de posição σx vira erro de tempo em cada linha: σt = σx / v, com v a velocidade local
    na linha, em pixels por segundo. As duas linhas são independentes. Mesma conta de
    src/orcamento_erro.py, seção 1.
    """
    sigmas = []
    for lado in ("entrada", "saida"):
        c = m[lado]
        f = dados["frame"]
        x0 = dados["cx"][f == c["frame_antes"]][0]
        x1 = dados["cx"][f == c["frame_depois"]][0]
        dt = dados["t"][f == c["frame_depois"]][0] - dados["t"][f == c["frame_antes"]][0]
        sigmas.append(m["sigma_x_px"] / (abs(x1 - x0) / dt))
    return math.hypot(*sigmas) / m["t_trecho"]


def erro_calibracao(cal):
    """Cada clique erra até meia resolução de clique; duas linhas: √2. Mesma conta do orçamento."""
    resolucao = cal.get("resolucao_clique_px", 1.0)
    distancia_px = abs(cal["x_linhas_px"][1] - cal["x_linhas_px"][0])
    return math.sqrt(2) * (resolucao / 2) / distancia_px


# =============================================================================
# VELOCIDADE INSTANTÂNEA
# =============================================================================

def desvio_da_tendencia(t, v):
    """Desvio padrão da velocidade em torno da sua média móvel de JANELA_TENDENCIA_S."""
    tendencia = np.array([v[np.abs(t - ti) <= JANELA_TENDENCIA_S / 2].mean() for ti in t])
    return float(np.std(v - tendencia, ddof=1))


def imprimir_instantanea(m):
    k = m["kalman"]
    pares = [a for a in m["instantaneas"] if abs(a["v_projetada"]) < LIMITE_PAR_GROSSEIRO]
    t_b = np.array([a["t"] for a in pares])
    v_b = np.array([a["v_projetada"] for a in pares])

    no_trecho = (k["t"] >= k["t_entrada_rts"]) & (k["t"] <= k["t_saida_rts"])
    t_s, v_s = k["t"][no_trecho], k["v_rts"][no_trecho]

    dt = float(np.median(np.diff(k["t"])))
    ruido_bruta = math.sqrt(2) * m["sigma_x_px"] * m["metros_por_pixel"] / dt
    disp_b, disp_s = desvio_da_tendencia(t_b, v_b), desvio_da_tendencia(t_s, v_s)

    print("\nVELOCIDADE INSTANTÂNEA (sem referência quadro a quadro: não é erro medido)")
    print(f"  ruído teórico da bruta, √2·σx·C/Δt: {ruido_bruta:.3f} m/s")
    print(f"  desvio previsto: filtro {k['dp_v_filtro'][no_trecho].mean():.3f} m/s | "
          f"suavizador {k['dp_medio_v_rts']:.3f} m/s (faixa de ±{2 * k['dp_medio_v_rts']:.3f} m/s)")
    print(f"  faixa de valores: bruta {v_b.min():.2f} a {v_b.max():.2f} m/s | "
          f"suavizada {k['v_min_rts']:.2f} a {k['v_max_rts']:.2f} m/s")
    print(f"  dispersão em torno da tendência de {JANELA_TENDENCIA_S:g} s: bruta {disp_b:.3f} m/s | "
          f"suavizada {disp_s:.3f} m/s | redução de {disp_b / disp_s:.2f} vezes")
    print("  (dispersão mistura ruído e oscilação real da passada)")


# =============================================================================
# RELATÓRIO POR ENSAIO
# =============================================================================

def avaliar(rotulo, nome):
    cal = json.loads((RAIZ / "calibracao" / f"{nome}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(ac.csv_maior_imgsz(nome))
    ref = ac.referencia_cronometro(nome)
    m = ac.medir(dados, cal, ref, com_kalman=True)

    print("\n" + "=" * 78)
    print(f"Ensaio {rotulo}: {nome}, trecho de {m['distancia']:g} m")
    print("=" * 78)
    print(f"10 tempos: {ref['tempos']}")

    ac.imprimir_comparacao(m)

    u_ruido = erro_ruido_posicao(dados, m)
    u_cal = erro_calibracao(cal)
    print("\nCONTRIBUIÇÃO DO PROCESSAMENTO PARA A INCERTEZA DA VELOCIDADE MÉDIA (não é o erro do sistema)")
    print(f"  ruído de posição nos cruzamentos: {100 * u_ruido:.3f}% | calibração: {100 * u_cal:.3f}% | "
          f"raiz da soma dos quadrados: {100 * math.hypot(u_ruido, u_cal):.3f}%")

    imprimir_instantanea(m)

    return rotulo, m


def imprimir_resumo(resultados):
    print("\n" + "=" * 78)
    print("RESUMO: COMPARAÇÃO COM O CRONÔMETRO")
    print("=" * 78)
    print(f"{'ensaio':>6} {'':<11}{'V sistema':>10}{'V crono':>9}{'t médio':>9}{'s':>8}{'erro bruto':>12}"
          f"{'em m/s':>9}{'U_ref':>7}{'U_s':>7}{'E_n':>7}{'compatível?':>13}")
    for rotulo, m in resultados:
        for nome, chave in (("sem filtro", "sem_filtro"), ("com filtro", "com_filtro")):
            c = m["comparacao"][chave]
            print(f"{rotulo:>6} {nome:<11}{c['v_sistema']:>10.4f}{c['v_ref']:>9.4f}{c['t_medio']:>9.3f}{c['s']:>8.3f}"
                  f"{c['erro_bruto']:>+11.2f}%{c['erro_bruto_ms']:>+9.4f}{c['U_ref_v']:>6.2f}%"
                  f"{c['U_s_v']:>6.2f}%{c['E_n']:>7.2f}{'sim' if c['compativel'] else 'NÃO':>13}")
    print("\nO erro bruto supõe cronômetro exato.")
    print("O erro do sistema não pode ser isolado com precisão melhor que a da referência.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    imprimir_resumo([avaliar(rotulo, nome) for rotulo, nome in ENSAIOS.items()])


if __name__ == "__main__":
    main()
