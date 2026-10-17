"""
Sensibilidade ao limite do teste de inovação do Kalman (somente leitura: só imprime).

Para cada limite (3σ, 4σ, 5σ, 10σ e sem teste), roda analisar_cruzamentos.medir() nos
dois ensaios de 20 m e mostra os quadros rejeitados, a velocidade média no trecho pelo
Kalman suavizado e o NIS médio do J_Variando. É a tabela de sensibilidade da monografia;
resultado e discussão em experimentos/2026-10-07-kalman.md (seções 4.1 e 8).

O limite padrão (kalman.LIMIAR_INOVACAO, 5σ) não muda: o script troca só o valor padrão
do argumento `limiar` de kalman.filtrar, que é por onde todas as chamadas passam.

Uso (na pasta do projeto):  python src/sensibilidade_limiar.py
"""
import json
import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
import analisar_cruzamentos as ac  # noqa: E402
import kalman  # noqa: E402

ENSAIOS = {"J_Correndo": 3.48, "J_Variando": 7.23}
LIMITES = (9.0, 16.0, 25.0, 100.0, math.inf)   # NIS; inf = sem teste


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dados = {n: ac.carregar_keypoints(RAIZ / f"resultados/keypoints/{n}_imgsz1920.csv") for n in ENSAIOS}
    cal = {n: json.loads((RAIZ / f"calibracao/{n}.json").read_text(encoding="utf-8")) for n in ENSAIOS}
    padrao = kalman.filtrar.__defaults__

    print(f"{'limite':>10} | {'rejeitados J_Correndo':>22} | {'rejeitados J_Variando':>34} | "
          f"{'V média Kalman J_Var.':>21} | {'NIS J_Var.':>10}")
    try:
        for limite in LIMITES:
            kalman.filtrar.__defaults__ = (limite,)
            k = {n: ac.medir(dados[n], cal[n], t, com_kalman=True)["kalman"] for n, t in ENSAIOS.items()}
            rotulo = "sem teste" if math.isinf(limite) else f"{math.sqrt(limite):g}σ ({limite:g})"
            print(f"{rotulo:>10} | {str(k['J_Correndo']['frames_rejeitados']):>22} | "
                  f"{str(k['J_Variando']['frames_rejeitados']):>34} | "
                  f"{k['J_Variando']['v_media_rts']:>21.4f} | "
                  f"{k['J_Variando']['consistencia']['nis_medio']:>10.2f}")
    finally:
        kalman.filtrar.__defaults__ = padrao


if __name__ == "__main__":
    main()
