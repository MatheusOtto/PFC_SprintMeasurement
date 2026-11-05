"""
Orçamento de incerteza da velocidade média (GUM, JCGM 100:2008), U_ref e E_n.
Só imprime. Uso (na raiz do repositório):  python src/orcamento_incerteza.py

  v = D/t  =>  u_rel(v)^2 = u_rel(D)^2 + u_rel(t)^2
  U = k * u_c, k = 2
  E_n = |delta| / sqrt(U_s^2 + U_ref^2), compatível se E_n <= 1
  U_s: só o que não é comum com a referência (ruído e cliques); D se cancela em delta.
  Incerteza do sistema: ruído, cliques e D.

Entradas lidas do projeto: calibracao/*.json, resultados/keypoints, resultados/cronometragens.csv.
Hipóteses marcadas com PARAMETRO: não são dado de campo.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import analisar_cruzamentos as ac  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ENSAIOS = {"A": ("Trotando_2", 30), "B": ("J_Correndo", 60), "C": ("J_Variando", 60)}
ENSAIOS_COM_GARRAFA = "BC"   # no Ensaio A, os marcadores foram chinelos, de tamanho não registrado
R3, R6 = math.sqrt(3), math.sqrt(6)

# --- PARAMETROS (hipóteses declaradas, faltam dados de campo) -----------------------------------
MARCADOR_M = 0.10          # garrafa de 10 cm: semilargura a = 5 cm por marcador, retangular
TRENA_M = 0.0              # trena: erro abaixo de 1 mm, desprezado; ver a sensibilidade abaixo
SEMI_HETZLER_NORMAL = 0.15 * math.sqrt(math.pi / 2)   # MAE de normal: MAE = sigma*sqrt(2/pi)
SEMI_HETZLER_RET = 0.15 * 2 / R3                      # MAE de retangular +-a: MAE = a/2, u = a/sqrt(3)
SEMI_HETZLER_RMS = math.hypot(0.15, 0.20)             # RMS do erro absoluto 0,15 +- 0,20 s


def crossing_term(dados, m):
    """u_t de cada linha: exata (com a fração da interpolação) e o limite sigma_x/v_local."""
    saida = {}
    for lado in ("entrada", "saida"):
        c = m[lado]
        f = dados["frame"]
        x0 = float(dados["cx"][f == c["frame_antes"]][0])
        x1 = float(dados["cx"][f == c["frame_depois"]][0])
        t0 = float(dados["t"][f == c["frame_antes"]][0])
        t1 = float(dados["t"][f == c["frame_depois"]][0])
        dt = t1 - t0
        v_px = abs(x1 - x0) / dt
        linha = x_linha(m, lado, x0, x1)
        alfa = (linha - x0) / (x1 - x0)
        limite = m["sigma_x_px"] / v_px
        exata = limite * math.hypot(1 - alfa, alfa)
        saida[lado] = dict(v_ms=v_px * m["metros_por_pixel"], alfa=alfa, limite=limite, exata=exata, dt=dt)
    return saida


def x_linha(m, lado, x0, x1):
    """A linha cruzada é a que fica entre x0 e x1."""
    for x in m["_x_linhas"]:
        if min(x0, x1) <= x <= max(x0, x1):
            return x
    raise ValueError(lado)


def conferir_pts(dados):
    dts = np.diff(dados["t"])
    return dict(med=float(np.median(dts)), min=float(dts.min()), max=float(dts.max()), dp=float(dts.std(ddof=1)))


def carimbos_no_trecho(dados, m):
    """Intervalos entre carimbos consecutivos dentro do trecho medido."""
    f = dados["frame"]
    janela = (f >= m["entrada"]["frame_antes"]) & (f <= m["saida"]["frame_depois"])
    dts = np.diff(dados["t"][janela])
    return dict(n=len(dts), med=float(np.median(dts)), min=float(dts.min()), max=float(dts.max()),
                dp=float(dts.std(ddof=1)), soma=float(dts.sum()))


res = {}
for rot, (nome, f) in ENSAIOS.items():
    cal = json.loads((RAIZ / "calibracao" / f"{nome}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(ac.csv_maior_imgsz(nome))
    ref = ac.referencia_cronometro(nome)
    m = ac.medir(dados, cal, ref, com_kalman=True)
    m["_x_linhas"] = cal["x_linhas_px"]
    D, T = m["distancia"], m["t_trecho"]
    C = m["metros_por_pixel"]

    # 1. ruído de posição nos cruzamentos
    cruz = crossing_term(dados, m)
    u_t1_lim = math.hypot(cruz["entrada"]["limite"], cruz["saida"]["limite"])
    u_t1_ex = math.hypot(cruz["entrada"]["exata"], cruz["saida"]["exata"])
    u_t1_vbar = math.sqrt(2) * m["sigma_x_px"] * C / (D / T)      # conta aproximada com v média

    # 2. cliques: cada um retangular +-(1 pixel de tela); a diferença de dois é triangular +-2a
    a_clique = cal.get("resolucao_clique_px", 1.0)
    px = cal["distancia_px"]
    u_clique_px = a_clique * math.sqrt(2) / R3        # igual a (2a)/sqrt(6), triangular de +-2a
    assert abs(u_clique_px - 2 * a_clique / R6) < 1e-12
    u_rel_clique = u_clique_px / px
    u_rel_clique_1px = (1 * math.sqrt(2) / R3) / px

    # 3. D e marcadores
    u_marc = math.sqrt(2) * (MARCADOR_M / 2) / R3
    u_D = math.hypot(u_marc, TRENA_M)
    u_rel_D = u_D / D

    u_rel_ruido = u_t1_lim / T

    # parcela que NÃO é comum com a referência: entra em E_n
    u_nc = math.hypot(u_rel_ruido, u_rel_clique)
    # incerteza do sistema: essa parcela mais D
    u_sistema = math.hypot(u_nc, u_rel_D)

    # B. referência
    t_med, s_t = ref["t_medio"], ref["s"]
    u_A = s_t / math.sqrt(10)
    U_ref = lambda u_H, u_A=u_A, t_med=t_med: 2 * math.hypot(u_A, u_H) / t_med   # noqa: E731
    U_ref_antigo = (2 * u_A + 0.15) / t_med

    c = m["comparacao"]
    res[rot] = dict(nome=nome, D=D, T=T, f=f, C=C, px=px, sigma_x=m["sigma_x_px"], a_clique=a_clique,
                    cruz=cruz, u_t1_lim=u_t1_lim, u_t1_ex=u_t1_ex, u_t1_vbar=u_t1_vbar,
                    u_rel_ruido=u_rel_ruido, u_rel_clique=u_rel_clique, u_rel_clique_1px=u_rel_clique_1px,
                    u_D=u_D, u_rel_D=u_rel_D, u_nc=u_nc,
                    u_sistema=u_sistema, t_med=t_med, s_t=s_t, u_A=u_A,
                    U_ref=U_ref, U_ref_antigo=U_ref_antigo,
                    v_s=c["sem_filtro"]["v_sistema"], v_f=c["com_filtro"]["v_sistema"], v_c=c["sem_filtro"]["v_ref"],
                    d_s=c["sem_filtro"]["erro_bruto"], d_f=c["com_filtro"]["erro_bruto"],
                    t_rts=m["kalman"]["t_trecho_rts"], pts=conferir_pts(dados), pts_trecho=carimbos_no_trecho(dados, m), base=dados["base_tempo"])


def pct(x, n=3):
    return f"{100 * x:.{n}f}%"


print("=" * 90 + "\nA. ORÇAMENTO DA VELOCIDADE MÉDIA\n" + "=" * 90)
for rot, r in res.items():
    print(f"\nEnsaio {rot} ({r['nome']}): D={r['D']:g} m, t={r['T']:.4f} s, f={r['f']} fps, "
          f"C={r['C'] * 1e3:.3f} mm/px, x_ref={r['px']:g} px, sigma_x={r['sigma_x']:.3f} px, sigma_x*C={r['sigma_x'] * r['C'] * 1e3:.2f} mm")
    print(f"  base de tempo: {r['base']}; dt mediano={r['pts']['med'] * 1e3:.3f} ms, min={r['pts']['min'] * 1e3:.3f}, "
          f"max={r['pts']['max'] * 1e3:.3f}, dp={r['pts']['dp'] * 1e3:.3f} ms")
    p = r["pts_trecho"]
    print(f"  carimbos no trecho: {p['n']} intervalos, mediana={p['med'] * 1e3:.3f} ms, min={p['min'] * 1e3:.3f}, "
          f"max={p['max'] * 1e3:.3f}, dp={p['dp'] * 1e3:.3f} ms; 1/f={1e3 / r['f']:.3f} ms")
    for lado, c in r["cruz"].items():
        print(f"  {lado:8s} v_local={c['v_ms']:.3f} m/s, fração={c['alfa']:.3f}, sigma_x/v={c['limite'] * 1e3:.2f} ms, "
              f"com a fração={c['exata'] * 1e3:.2f} ms")
    print(f"  1. ruído: 2 linhas={r['u_t1_lim'] * 1e3:.2f} ms => {pct(r['u_rel_ruido'])} "
          f"(com a fração {r['u_t1_ex'] * 1e3:.2f} ms, {pct(r['u_t1_ex'] / r['T'])}; "
          f"conta com v média {r['u_t1_vbar'] * 1e3:.2f} ms, {pct(r['u_t1_vbar'] / r['T'])})")
    print(f"  2. cliques: +-{r['a_clique']} px => {pct(r['u_rel_clique'])} (com +-1 px: {pct(r['u_rel_clique_1px'])})")
    print(f"  U_s (1,2): u={pct(r['u_nc'])}, U_s={pct(2 * r['u_nc'], 2)}")
    if rot in ENSAIOS_COM_GARRAFA:
        print(f"  3. D+marcadores (a={MARCADOR_M / 2 * 100:.0f} cm, trena={TRENA_M}): u(D)={r['u_D'] * 100:.2f} cm => {pct(r['u_rel_D'])}")
        print(f"  sistema (1,2,3): u={pct(r['u_sistema'])}, U={pct(2 * r['u_sistema'], 2)}")

print("\nSensibilidade à trena (u trena em cm => U do sistema), Ensaios B e C:")
for ut in (0, 0.1, 0.5, 1, 2, 5):
    linha = []
    for rot in "BC":
        r = res[rot]
        u_marc = math.sqrt(2) * (MARCADOR_M / 2) / R3
        uD = math.hypot(u_marc, ut / 100) / r["D"]
        linha.append(pct(2 * math.sqrt(r["u_nc"] ** 2 + uD**2), 2))
    print(f"  {ut:>3} cm: B {linha[0]}, C {linha[1]}")

print("\nSensibilidade ao tamanho do marcador (largura em cm => U do sistema), Ensaios B e C:")
for w in (5, 7.5, 10):
    linha = []
    for rot in "BC":
        r = res[rot]
        uD = math.sqrt(2) * (w / 100 / 2) / R3 / r["D"]
        linha.append(pct(2 * math.sqrt(r["u_nc"] ** 2 + uD**2), 2))
    print(f"  {w:>4} cm: B {linha[0]}, C {linha[1]}")

print("\nFiltragem na velocidade média (conferência do 0,06%):")
for rot, r in res.items():
    print(f"  {rot}: {r['v_s']:.4f} -> {r['v_f']:.4f} m/s = {pct((r['v_f'] - r['v_s']) / r['v_s'], 3)}; "
          f"t {r['T']:.4f} -> {r['t_rts']:.4f} s; delta {r['d_s']:+.2f} -> {r['d_f']:+.2f}")

print("\n" + "=" * 90 + "\nB. U_ref E E_n\n" + "=" * 90)
print(f"u_H: normal={SEMI_HETZLER_NORMAL:.3f} s; retangular={SEMI_HETZLER_RET:.3f} s; RMS (0,15;0,20)={SEMI_HETZLER_RMS:.3f} s")
cenarios = (("normal (adotada)", SEMI_HETZLER_NORMAL), ("retangular", SEMI_HETZLER_RET),
            ("RMS 0,25 s", SEMI_HETZLER_RMS), ("sem 0,15 s", 0.0))
for rot, r in res.items():
    print(f"\nEnsaio {rot}: t_med={r['t_med']:.3f} s, s_t={r['s_t']:.4f} s, u_A={r['u_A'] * 1e3:.1f} ms; "
          f"U_t antigo={pct(r['U_ref_antigo'], 2)}; delta sem={r['d_s']:+.2f}%, com={r['d_f']:+.2f}%; "
          f"t_sist-t_med={r['T'] - r['t_med']:+.3f} s")
    Us_nc = 2 * r["u_nc"] * 100
    for nome, u_H in cenarios:
        Ur = r["U_ref"](u_H) * 100
        en = [abs(d) / math.hypot(Us_nc, Ur) for d in (r["d_s"], r["d_f"])]
        print(f"  {nome:18s} u_H={u_H:.3f} s, U_ref={Ur:6.2f}% | E_n (U_s={Us_nc:.2f}%): sem {en[0]:.2f}, com {en[1]:.2f}")
    # limiar: u_H em que E_n = 1, de U_ref^2 = delta^2 - U_s^2 e U_ref = 2*sqrt(u_A^2 + u_H^2)/t_med
    for rotulo, d in (("sem", r["d_s"]), ("com", r["d_f"])):
        u_total2 = (r["t_med"] / 200) ** 2 * (d**2 - Us_nc**2)
        if u_total2 > r["u_A"] ** 2:
            print(f"  limiar de u_H ({rotulo} filtro): E_n = 1 em u_H = {math.sqrt(u_total2 - r['u_A'] ** 2):.4f} s")
        else:
            print(f"  limiar de u_H ({rotulo} filtro): não há, E_n < 1 mesmo com u_H = 0")

# Refazer as contas à mão, com os números impressos no texto (segunda conferência)
print("\n" + "=" * 90 + "\nSEGUNDA CONFERÊNCIA, SÓ COM NÚMEROS DO TEXTO\n" + "=" * 90)
print(f"B sigma_x*C = 0,698*13,298 = {0.698 * 13.298:.2f} mm; R = {(0.698 * 13.298e-3) ** 2:.3g} m2")
print(f"C R = (6,46 mm)^2 = {(6.46e-3) ** 2:.3g} m2")
print(f"B com v média: sqrt(2)*9,28/5,8669 = {math.sqrt(2) * 9.28 / 5.8669:.2f} ms; /3,409 s = {math.sqrt(2) * 9.28e-3 / 5.8669 / (20 / 5.8669) * 100:.3f}%")
print(f"C com v média: sqrt(2)*6,46/2,7634 = {math.sqrt(2) * 6.46 / 2.7634:.2f} ms; /7,237 s = {math.sqrt(2) * 6.46e-3 / 2.7634 / (20 / 2.7634) * 100:.3f}%")
print(f"C filtragem: (2,7650-2,7634)/2,7634 = {(2.7650 - 2.7634) / 2.7634 * 100:.3f}%")
print(f"B filtragem: (5,8625-5,8669)/5,8669 = {(5.8625 - 5.8669) / 5.8669 * 100:.3f}%")
for rot, s, t in (("A", 0.0295, 2.516), ("B", 0.0767, 3.399), ("C", 0.0469, 7.127)):
    uA = s / math.sqrt(10)
    print(f"{rot} U_ref(normal) = 2*sqrt({uA:.4f}^2+0,1880^2)/{t} = {2 * math.hypot(uA, 0.15 * math.sqrt(math.pi / 2)) / t * 100:.2f}%; sem 0,15: {2 * uA / t * 100:.2f}%")
