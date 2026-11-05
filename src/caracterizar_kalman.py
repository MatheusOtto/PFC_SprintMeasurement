"""
=============================================================================
CARACTERIZAÇÃO DO FILTRO DE KALMAN NOS ENSAIOS
=============================================================================

Somente leitura: só imprime, não grava nada. Para cada ensaio (CSV de keypoints de maior
imgsz), mostra de onde vêm R e q e as evidências sobre o filtro:

  1. ruído: segundas diferenças descartadas, σx pelo desvio padrão, σ pelo MAD, a razão entre
     os dois e a curtose de d; movimento em 3 quadros comparado com o ruído;
  2. histograma de d comparado com a curva normal, em tabela;
  3. sensibilidade de R e q ao corte dos erros grosseiros (4, 5 e 10 desvios);
  4. σx na primeira e na segunda metade do trecho (a hipótese é um único R por vídeo);
  5. média de e²/S × q, na varredura da escolha de q;
  6. evidências das inovações: autocorrelação, fração acima de 2 desvios, quadros rejeitados
     e maiores valores aceitos. A média de e²/S vale 1 por construção e não é evidência;
  7. sensibilidade ao desvio inicial da velocidade (5, 10 e 50 m/s);
  8. sensibilidade à margem antes e depois do trecho (0,25, 0,5 e 1,0 s);
  9. velocidade média com e sem filtro e comparação com o cronômetro.

Uso (na raiz do repositório):  python src/caracterizar_kalman.py
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

ENSAIOS = {"A": "Trotando_2", "B": "J_Correndo", "C": "J_Variando"}

CORTES = (4.0, 5.0, 10.0)
DESVIOS_INICIAIS = (5.0, 10.0, 50.0)
MARGENS = (0.25, 0.5, 1.0)

# Histograma de d, em desvios padrão dos d usados em R.
BORDAS_HISTOGRAMA = np.arange(-4.0, 4.5, 1.0)
ACELERACAO_EXEMPLO = 5.0       # m/s²: movimento em 3 quadros ≈ a·Δt²
PASSO_TABELA_Q = 4             # imprime um a cada 4 pontos da varredura
MAIORES_ACEITAS = 3


def fracao_normal(a, b):
    """Fração da curva normal padrão entre a e b."""
    return 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))


def imprimir_ruido(m, dt):
    C = m["metros_por_pixel"]
    ruido_d = math.sqrt(6) * m["sigma_x_px"] * C
    print("\n[1] Ruído da medição")
    print(f"   segundas diferenças: {m['n_trincas']}, descartadas {m['d_descartados']}")
    print(f"   σx pelo desvio padrão {m['sigma_x_px']:.4f} px = {100 * m['sigma_x_px'] * C:.3f} cm | "
          f"σ pelo MAD {m['sigma_mad_px']:.4f} px | razão {m['razao_desvio_mad']:.3f} | "
          f"curtose de d {m['curtose_d']:.2f} (normal: 3)")
    print(f"   R = (σx·C)² = {(m['sigma_x_px'] * C) ** 2:.4e} m²")
    print(f"   parte de ruído em d: √6·σx·C = {100 * ruido_d:.2f} cm | movimento a {ACELERACAO_EXEMPLO:g} m/s²: "
          f"a·Δt² = {1000 * ACELERACAO_EXEMPLO * dt ** 2:.2f} mm")


def imprimir_histograma(m):
    d, usados = m["d"], m["d_usados"]
    z = (d - usados.mean()) / usados.std(ddof=1)
    print("\n[2] Histograma de d, em desvios padrão dos d usados, contra a curva normal")
    print(f"   {'faixa':>18} {'observado':>10} {'normal':>8}")
    print(f"   {'abaixo de ' + format(BORDAS_HISTOGRAMA[0], '+.1f'):>18} "
          f"{int(np.sum(z < BORDAS_HISTOGRAMA[0])):>10} "
          f"{len(z) * fracao_normal(-math.inf, BORDAS_HISTOGRAMA[0]):>8.1f}")
    for a, b in zip(BORDAS_HISTOGRAMA[:-1], BORDAS_HISTOGRAMA[1:]):
        print(f"   {f'{a:+.1f} a {b:+.1f}':>18} {int(np.sum((z >= a) & (z < b))):>10} "
              f"{len(z) * fracao_normal(a, b):>8.1f}")
    print(f"   {'acima de ' + format(BORDAS_HISTOGRAMA[-1], '+.1f'):>18} "
          f"{int(np.sum(z >= BORDAS_HISTOGRAMA[-1])):>10} "
          f"{len(z) * fracao_normal(BORDAS_HISTOGRAMA[-1], math.inf):>8.1f}")


def imprimir_corte(dados, m):
    C = m["metros_por_pixel"]
    print("\n[3] Sensibilidade ao corte dos erros grosseiros em d")
    print(f"   {'corte':>6} {'descartados':>11} {'σx (px)':>8} {'R (m²)':>11} {'q (m²/s³)':>10} "
          f"{'V filtro':>9} {'fora/10':>8}  rejeitados")
    for corte in CORTES:
        r = ac.medir_ruido(m["d"], corte)
        R = (r["sigma_x_px"] * C) ** 2
        k = ac.aplicar_kalman(dados, m, R=R)
        print(f"   {corte:>6g} {r['d_descartados']:>11} {r['sigma_x_px']:>8.4f} {R:>11.4e} {k['q']:>10.3f} "
              f"{k['v_media_rts']:>9.4f} {k['verificacao']['atrasos_fora']:>8}  {k['frames_rejeitados']}")


def imprimir_metades(m):
    d = m["d"]
    meio = len(d) // 2
    print("\n[4] σx em cada metade do trecho (com o mesmo corte)")
    print(f"   1ª metade {ac.medir_ruido(d[:meio])['sigma_x_px']:.4f} px | "
          f"2ª metade {ac.medir_ruido(d[meio:])['sigma_x_px']:.4f} px | "
          f"trecho inteiro {m['sigma_x_px']:.4f} px")


def imprimir_curva_q(k):
    e = k["escolha_q"]
    print("\n[5] Média de e²/S × q (a escolha é onde ela cruza 1)")
    print(f"   {'q (m²/s³)':>10} {'média':>9}")
    for q, nis in list(zip(e["curva_q"], e["curva_nis"]))[::PASSO_TABELA_Q]:
        print(f"   {q:>10.3f} {'sem aceitas' if math.isnan(nis) else format(nis, '.3f'):>9}")
    print(f"   escolhido: q = {k['q']:.3f} m²/s³, média {e['nis_medio']:.4f}, "
          f"{'convergiu' if e['convergiu'] else 'NÃO atingiu a tolerância'}")


def imprimir_evidencias(dados, m, k):
    c = k["verificacao"]
    print("\n[6] Evidências das inovações, nas medições aceitas do trecho")
    print(f"   N = {c['N']} | limite da autocorrelação ±{c['limite_autocorrelacao']:.3f} | "
          f"atrasos fora do limite: {c['atrasos_fora']} de {kalman.ATRASO_MAX}")
    print("   autocorrelação, atrasos 1 a 10: "
          + " ".join(f"{r:+.3f}" for r in c["autocorrelacao"]))
    print(f"   fração de |e|/√S acima de 2: {100 * c['fracao_acima_2']:.1f}% (esperado perto de 5%)")

    # Valores de |e|/√S quadro a quadro, com o mesmo q, para mostrar o que ficou perto do limite.
    janela = np.isin(dados["frame"], k["frames"])
    z = m["direcao"] * dados["cx"][janela] * m["metros_por_pixel"]
    f = kalman.filtrar(k["t"], z, k["R"], k["q"])
    desvios = np.abs(f["e"]) / np.sqrt(f["S"])

    for i in np.flatnonzero(f["rejeitado"]):
        print(f"   rejeitado: quadro {k['frames'][i]}, {desvios[i]:.2f} desvios, e = {100 * f['e'][i]:+.2f} cm")
    if not f["rejeitado"].any():
        print("   nenhum quadro rejeitado")

    aceitos = np.where(f["rejeitado"], np.nan, desvios)
    ordem = np.argsort(np.nan_to_num(aceitos, nan=-1.0))[::-1][:MAIORES_ACEITAS]
    print("   maiores aceitos: " + ", ".join(f"quadro {k['frames'][i]} ({aceitos[i]:.2f})" for i in ordem))


def maior_diferenca(a, b):
    """Maior diferença de velocidade suavizada entre duas rodadas, nos quadros comuns do trecho."""
    comuns, ia, ib = np.intersect1d(a["frames"], b["frames"], return_indices=True)
    dentro = (a["t"][ia] >= a["t_entrada_rts"]) & (a["t"][ia] <= a["t_saida_rts"])
    return float(np.max(np.abs(a["v_rts"][ia] - b["v_rts"][ib])[dentro]))


def imprimir_inicializacao(dados, m, k):
    print("\n[7] Sensibilidade ao desvio inicial da velocidade")
    print(f"   {'desvio (m/s)':>12} {'q escolhido':>12} {'maior Δv com q refeito':>24} {'maior Δv com o mesmo q':>24}")
    for dp in DESVIOS_INICIAIS:
        refeito = ac.aplicar_kalman(dados, m, dp_inicial_v=dp)
        mesmo_q = ac.aplicar_kalman(dados, m, q=k["q"], dp_inicial_v=dp)
        print(f"   {dp:>12g} {refeito['q']:>12.3f} {maior_diferenca(k, refeito):>24.2e} "
              f"{maior_diferenca(k, mesmo_q):>24.2e}")


def imprimir_margem(dados, m, k):
    print("\n[8] Sensibilidade à margem antes e depois do trecho")
    print(f"   {'margem (s)':>10} {'1º quadro':>10} {'q escolhido':>12} {'V filtro':>9} {'maior Δv':>10}")
    for margem in MARGENS:
        r = ac.aplicar_kalman(dados, m, margem=margem)
        print(f"   {margem:>10g} {r['frames'][0]:>10} {r['q']:>12.3f} {r['v_media_rts']:>9.4f} "
              f"{maior_diferenca(k, r):>10.2e}")


def imprimir_velocidades(m, k, ref):
    print("\n[9] Velocidade média no trecho e cronômetro (assistindo ao vídeo)")
    print(f"   tempos: {ref['tempos']}")
    vk = k["v_media_rts"]
    print(f"   sem filtro {m['v_media']:.4f} m/s | com filtro {vk:.4f} m/s "
          f"({100 * (vk / m['v_media'] - 1):+.3f}% do sem filtro)")
    ac.imprimir_comparacao(m)


def caracterizar(rotulo, nome):
    csv_kp = ac.csv_maior_imgsz(nome)
    cal = json.loads((RAIZ / "calibracao" / f"{nome}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(csv_kp)
    ref = ac.referencia_cronometro(nome)
    m = ac.medir(dados, cal, ref, com_kalman=True)
    k = m["kalman"]
    dt = float(np.median(np.diff(k["t"])))

    print("\n" + "=" * 78)
    print(f"Ensaio {rotulo}: {nome} ({csv_kp.name})   {1 / dt:.2f} fps   "
          f"escala {m['metros_por_pixel'] * 100:.4f} cm/px")
    print("=" * 78)

    imprimir_ruido(m, dt)
    imprimir_histograma(m)
    imprimir_corte(dados, m)
    imprimir_metades(m)
    imprimir_curva_q(k)
    imprimir_evidencias(dados, m, k)
    imprimir_inicializacao(dados, m, k)
    imprimir_margem(dados, m, k)
    imprimir_velocidades(m, k, ref)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for rotulo, nome in ENSAIOS.items():
        caracterizar(rotulo, nome)


if __name__ == "__main__":
    main()
