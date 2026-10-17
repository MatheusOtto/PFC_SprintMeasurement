"""
Orçamento de erro empírico dos ensaios (somente leitura: só imprime, não grava nada).
Para cada ensaio usa o CSV de keypoints de maior imgsz (a resolução nativa).

Uso (na pasta do projeto):  python src/orcamento_erro.py
Resultados e ressalvas: diagnostico/orcamento_erro.md

0. base de tempo (resíduo dos carimbos) e calibração por cliques (meia resolução de
   clique, lida do JSON da calibração)
1. ruído de tempo nos cruzamentos (analítico, com o σz do Kalman)
2. efeito do FPS: decimação 60 -> 30/20/15/10 fps, todas as fases, com e sem interpolação
3. ponto do corpo usado como referência (ombros, cada ombro, quadril, nariz, bbox)
4. precisão da velocidade média numa janela de duração T
5. FPS x Kalman (CV) na velocidade instantânea
6. rolling shutter (linha dos ombros na entrada x saída)
7. profundidade (corredor fora do plano das marcas)
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import analisar_cruzamentos as ac  # noqa: E402
import kalman  # noqa: E402

ENSAIOS = {"Trotando_2": 2.55, "J_Correndo": 3.48, "J_Variando": 7.23}
CHAVES = ("frame", "t", "t_nominal", "t_carimbo", "n_pessoas", "pessoa", "cx", "ombros", "bbox")


def csv_maior_imgsz(nome):
    """CSV de keypoints do ensaio com o maior imgsz (a resolução nativa, padrão do pipeline)."""
    candidatos = []
    for c in (RAIZ / "resultados" / "keypoints").glob(f"{nome}_imgsz*.csv"):
        sufixo = c.stem.rpartition("_imgsz")[2]
        if sufixo.isdigit():
            candidatos.append((int(sufixo), c))
    return max(candidatos)[1]


def ler_colunas(caminho):
    with open(caminho, encoding="utf-8-sig") as f:
        linhas = list(csv.reader(f, delimiter=";"))
    cab = linhas[0]

    def col(nome):
        i = cab.index(nome)
        return np.array([float(l[i]) if l[i] != "" else np.nan for l in linhas[1:]])

    def kp(k):
        nx = next(n for n in cab if n.startswith(f"kp{k}_") and n.endswith("_x"))
        base = nx[:-2]
        return col(base + "_x"), col(base + "_y"), col(base + "_conf")

    return col, kp


def subconjunto(dados, mascara):
    novo = dict(dados)
    for c in CHAVES:
        if dados.get(c) is not None:
            novo[c] = dados[c][mascara]
    return novo


def com_cx(dados, cx):
    novo = dict(dados)
    novo["cx"] = cx
    return novo


def v_sem_interp(dados, m):
    """Cruzamento = primeiro frame depois da linha (quantização pura do FPS)."""
    t = dados["t"]
    f = dados["frame"]
    t_e = t[f == m["entrada"]["frame_depois"]][0]
    t_s = t[f == m["saida"]["frame_depois"]][0]
    return m["distancia"] / (t_s - t_e)


def ajuste_janela(t, z, T):
    """Inclinação por regressão em janelas não sobrepostas de duração T; devolve o EP mediano."""
    eps = []
    inicio = t[0]
    while inicio + T <= t[-1]:
        sel = (t >= inicio) & (t < inicio + T)
        if sel.sum() >= 3:
            tt, zz = t[sel], z[sel]
            A = np.column_stack([tt - tt.mean(), np.ones_like(tt)])
            coef, res, *_ = np.linalg.lstsq(A, zz, rcond=None)
            resid = zz - A @ coef
            s2 = resid @ resid / (len(tt) - 2)
            eps.append(np.sqrt(s2 / np.sum((tt - tt.mean()) ** 2)))
        inicio += T
    return np.median(eps) if eps else np.nan, len(eps)


for nome, t_crono in ENSAIOS.items():
    csv_kp = csv_maior_imgsz(nome)
    meta = json.loads(csv_kp.with_suffix(".json").read_text(encoding="utf-8"))
    cal = json.loads((RAIZ / f"calibracao/{nome}.json").read_text(encoding="utf-8"))
    dados = ac.carregar_keypoints(csv_kp)
    m = ac.medir(dados, cal, t_crono, com_kalman=True)
    k = m["kalman"]
    mpp = m["metros_por_pixel"]
    fps = 1.0 / np.median(np.diff(dados["t"]))

    print("\n" + "=" * 78)
    print(f"{nome} ({csv_kp.name})   V média = {m['v_media']:.4f} m/s   T = {m['t_trecho']:.4f} s "
          f"({m['base_tempo']})   crono = {t_crono} s ({m['erro_rel']:+.2f}%)")
    print(f"fps = {fps:.2f}   escala = {mpp*1000:.2f} mm/px   σz (Kalman) = {k['sigma_z_px']:.2f} px "
          f"= {k['sigma_z_px']*mpp*1000:.1f} mm")
    print(f"V pela base nominal (quadro / FPS do contêiner) = {m['v_nominal']:.4f}  "
          f"({100*(m['v_nominal']/m['v_media']-1):+.3f}% vs base oficial)")

    # ---------------------------------------------------------------- 0. base de tempo e calibração
    print("\n[0] Base de tempo e calibração")
    if dados["t_carimbo"] is not None:
        intervalos = np.diff(dados["t_carimbo"])[1:]   # sem o 1º intervalo, anômalo nos vídeos a 60 fps
        dp_int = float(np.std(intervalos, ddof=1))
        # O tempo do trecho é a diferença de dois carimbos: desvio √2 · dp dos intervalos (cota superior).
        print(f"   carimbos: intervalo médio {1000*intervalos.mean():.4f} ms ({1/intervalos.mean():.3f} fps), "
              f"desvio {1000*dp_int:.3f} ms -> σ(T)/T ≤ {100*np.sqrt(2)*dp_int/m['t_trecho']:.4f}%")
    res_clique = cal.get("resolucao_clique_px", 1.0)
    dist_px = abs(cal["x_linhas_px"][1] - cal["x_linhas_px"][0])
    # Cada clique erra até meia resolução de clique; duas linhas: √2.
    print(f"   calibração: resolução de clique {res_clique:.2f} px, {dist_px:.0f} px entre as linhas "
          f"-> σ(C)/C ≈ √2 · {res_clique/2:.2f} / {dist_px:.0f} = {100*np.sqrt(2)*res_clique/2/dist_px:.3f}%")

    # ---------------------------------------------------------------- 1. ruído de tempo
    print("\n[1] Ruído de tempo nos cruzamentos (σt = σz / v_local)")
    sig = []
    for lado in ("entrada", "saida"):
        c = m[lado]
        f = dados["frame"]
        x0 = dados["cx"][f == c["frame_antes"]][0]
        x1 = dados["cx"][f == c["frame_depois"]][0]
        dt = dados["t"][f == c["frame_depois"]][0] - dados["t"][f == c["frame_antes"]][0]
        v_px = abs(x1 - x0) / dt
        s = k["sigma_z_px"] / v_px
        sig.append(s)
        print(f"   {lado:8s}: v local = {v_px*mpp:5.2f} m/s ({abs(x1-x0):.1f} px/frame)  σt = {s*1000:.2f} ms")
    s_dt = np.hypot(*sig)
    print(f"   σ(Δt) = {s_dt*1000:.2f} ms  ->  σ(V)/V = {100*s_dt/m['t_trecho']:.3f}%  "
          f"= {m['v_media']*s_dt/m['t_trecho']*1000:.1f} mm/s")
    q = 1 / fps
    print(f"   referência: só quantização do frame (sem interpolar): σ(Δt) = {q/np.sqrt(6)*1000:.2f} ms "
          f"({100*q/np.sqrt(6)/m['t_trecho']:.3f}%), pior caso ±{q*1000:.1f} ms ({100*q/m['t_trecho']:.2f}%)")
    print(f"   Kalman suavizado vs interpolação: {100*(k['v_media_rts']/m['v_media']-1):+.3f}%")

    # ---------------------------------------------------------------- 2. FPS
    print(f"\n[2] Decimação de FPS (todas as fases). Diferença da V média para o resultado a {fps:.0f} fps")
    print(f"   {'fps':>5} {'fases':>5} | {'interp: média':>13} {'dp':>7} {'máx|Δ|':>7} | "
          f"{'sem interp: dp':>14} {'máx|Δ|':>7}")
    v_ref = m["v_media"]
    m60_sem = 100 * (v_sem_interp(dados, m) / v_ref - 1)
    print(f"   {fps:>5.0f} {1:>5} | {0:>+13.3f} {0:>7.3f} {0:>7.3f} | {'(1 fase)':>14} {abs(m60_sem):>7.3f}")
    idx = np.arange(len(dados["frame"]))
    for passo in (2, 3, 4, 6):
        d_int, d_sem = [], []
        for fase in range(passo):
            sub = subconjunto(dados, idx % passo == fase)
            ms = ac.medir(sub, cal, com_kalman=False)
            if ms["saida"] is None:
                continue
            d_int.append(100 * (ms["v_media"] / v_ref - 1))
            d_sem.append(100 * (v_sem_interp(sub, ms) / v_ref - 1))
        d_int, d_sem = np.array(d_int), np.array(d_sem)
        print(f"   {fps/passo:>5.0f} {len(d_int):>5} | {d_int.mean():>+13.3f} {d_int.std(ddof=1):>7.3f} "
              f"{np.abs(d_int).max():>7.3f} | {d_sem.std(ddof=1):>14.3f} {np.abs(d_sem).max():>7.3f}")

    # ---------------------------------------------------------------- 3. ponto do corpo
    print("\n[3] Ponto do corpo usado como referência (mesmas linhas, mesmo critério de confiança)")
    col, kp = ler_colunas(csv_kp)

    def ponto(indices):
        xs, ok = [], np.ones(len(dados["frame"]), bool)
        for i in indices:
            x, y, c = kp(i)
            with np.errstate(invalid="ignore"):
                ok &= (x > 0) & (y > 0) & (c >= ac.CONFIANCA_MIN_OMBRO)
            xs.append(x)
        return np.where(ok, np.mean(xs, axis=0), np.nan)

    bx = (col("bbox_x1") + col("bbox_x2")) / 2
    pontos = {
        "centro dos ombros (5,6)": ponto([5, 6]),
        "ombro esquerdo (5)": ponto([5]),
        "ombro direito (6)": ponto([6]),
        "centro do quadril (11,12)": ponto([11, 12]),
        "nariz (0)": ponto([0]),
        "centro da bbox": bx,
    }
    for rotulo, cx in pontos.items():
        mp = ac.medir(com_cx(dados, cx), cal, com_kalman=False)
        if mp["saida"] is None:
            print(f"   {rotulo:28s}: sem cruzamento")
            continue
        sx = mp["sigma_x_px"]
        print(f"   {rotulo:28s}: V = {mp['v_media']:.4f}  ({100*(mp['v_media']/v_ref-1):+.2f}%)  "
              f"σx = {sx:.2f} px  disp = {mp['disponibilidade']:.0f}%")

    # ---------------------------------------------------------------- 4. janela
    print("\n[4] Precisão da velocidade média numa janela T (regressão linear; EP mediano)")
    t = dados["t"]
    sel = (t >= m["entrada"]["t"]) & (t <= m["saida"]["t"]) & ~np.isnan(dados["cx"])
    tt = t[sel]
    zz = m["direcao"] * dados["cx"][sel] * mpp
    sz = k["sigma_z_px"] * mpp
    print(f"   {'T (s)':>7} {'n':>4} | {'EP empírico':>11} {'σ teórico (ruído)':>17}")
    print(f"   {1/fps:>7.3f} {'-':>4} | {'(bruta)':>11} {np.sqrt(2)*sz*fps:>17.3f}")
    for T in (0.1, 1 / 6, 0.25, 0.286, 0.5, 1.0):
        ep, n = ajuste_janela(tt, zz, T)
        N = T * fps
        # N pontos igualmente espaçados: σ da inclinação = σz / sqrt(Σ(t − t̄)²)
        tk = np.arange(int(round(N))) / fps
        teor = sz / np.sqrt(np.sum((tk - tk.mean()) ** 2))
        print(f"   {T:>7.3f} {n:>4} | {ep:>11.3f} {teor:>17.3f}")

    # ---------------------------------------------------------------- 5. FPS x Kalman
    print("\n[5] FPS x Kalman (CV, r do ensaio, q reajustado) — velocidade instantânea no trecho")
    cv60 = k
    t60, v60 = cv60["t"], cv60["v_rts"]
    janela = ((t >= m["entrada"]["t"] - ac.MARGEM_KALMAN_S)
              & (t <= m["saida"]["t"] + ac.MARGEM_KALMAN_S))
    print(f"   {'fps':>5} | {'±σ RTS mediano':>14} {f'RMS vs {fps:.0f} fps':>13} {'V méd. RTS':>10}")
    for passo in (1, 2, 4):
        mask = janela & (idx % passo == 0)
        tj = t[mask]
        zj = m["direcao"] * dados["cx"][mask] * mpp
        res = kalman.estimar(tj, zj, k["r"], kalman.ModeloCV)
        no = (tj >= m["entrada"]["t"]) & (tj <= m["saida"]["t"])
        z_e = m["direcao"] * m["entrada"]["x_linha"] * mpp
        z_s = m["direcao"] * m["saida"]["x_linha"] * mpp
        te = ac.cruzamento_continuo(tj, res["z_rts"], z_e)
        ts = ac.cruzamento_continuo(tj, res["z_rts"], z_s, te)
        rms = np.sqrt(np.mean((res["v_rts"][no] - np.interp(tj[no], t60, v60)) ** 2))
        print(f"   {fps/passo:>5.0f} | {np.median(res['dp_v_rts'][no]):>14.3f} {rms:>13.3f} "
              f"{m['distancia']/(ts-te):>10.4f}   q = {res['ajuste']['params']['q']:.2f}")

    # ---------------------------------------------------------------- 6. rolling shutter
    print("\n[6] Linha (y) dos ombros nos cruzamentos — rolling shutter")
    ys = []
    for lado in ("entrada", "saida"):
        f = dados["frame"] == m[lado]["frame_depois"]
        o = dados["ombros"][f][0]
        ys.append((o[1] + o[3]) / 2)
    dy = ys[1] - ys[0]
    print(f"   y entrada = {ys[0]:.0f} px, y saída = {ys[1]:.0f} px, Δy = {dy:+.0f} px")
    for leitura_ms in (10, 20, 30):
        dt = dy / meta["altura_px"] * leitura_ms / 1000
        print(f"   leitura do sensor {leitura_ms} ms -> Δt = {dt*1000:+.2f} ms ({100*dt/m['t_trecho']:+.3f}%)")

print("\n[7] Profundidade (corredor fora do plano das marcas, câmera a 21 m)")
for d in (0.1, 0.25, 0.5, 1.0):
    print(f"   {d:4.2f} m fora do plano -> erro de escala {100*d/21:.2f}%")
