# Medição de velocidade de corrida por visão computacional

O sistema estima a velocidade de um corredor a partir do vídeo de uma câmera fixa em vista lateral. Em cada quadro, uma rede YOLO de pose localiza os ombros, e o centro entre eles é o ponto medido. O resultado é a velocidade média no trecho e a curva de velocidade instantânea, obtida com um filtro e um suavizador de Kalman. A velocidade média é comparada com a cronometragem manual.

## Instalação

É preciso ter o Python 3.14. No Windows, na raiz do repositório:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Rodar o sistema

```
python src/visualizar_medicao.py
```

O programa faz três perguntas no terminal. Enter escolhe a opção 1.

1. **Vídeo:** `J_Correndo` ou `J_Variando`.
2. **Calibração:** usar a padrão ou marcar os pontos no chão.
3. **Cronômetro:** usar os tempos gravados, cuja média é mostrada, ou digitar outros, separados por espaço.

Para marcar os pontos, informe a distância real entre eles e clique nas duas marcações no solo. A lupa no canto da janela ajuda a acertar o ponto. `R` refaz os pontos, `C` confirma e `Q` cancela. A marcação é gravada em `saidas/calibracao/`, e a calibração padrão não muda. Marcar errado de propósito mostra como o erro afeta a medição.

Depois, o relatório aparece no terminal e o vídeo é reproduzido com a medição sobreposta. No fim, a tela mostra os resultados e a comparação com o cronômetro.

| Tecla | Função |
|---|---|
| `ESPAÇO` | pausar ou continuar |
| `+` e `-` | mais rápido ou mais lento |
| `R` | recomeçar |
| `S` | gravar a tela em `saidas/figuras/` |
| `Q` | sair |

## Reproduzir os resultados

Estes comandos só usam os dados do repositório e imprimem os números no terminal:

```
python src/erros_velocidade.py        # comparação com o cronômetro nos três ensaios
python src/caracterizar_kalman.py     # parâmetros do filtro em cada ensaio
python src/orcamento_erro.py          # ruído dos pontos candidatos e demais fontes de erro
python src/orcamento_incerteza.py     # incerteza da velocidade média, U_ref e E_n
python src/segmentacao_hsv.py         # segmentação por cor no Ensaio A
python src/calculo_suavizador.py      # cálculo do suavizador em um quadro do Ensaio C
python src/analisar_cruzamentos.py resultados/keypoints/J_Correndo_imgsz1920.csv --calibracao calibracao/J_Correndo.json
```

Para o Ensaio A com o quadro reduzido a 640 pixels, use `resultados/keypoints/Trotando_2_imgsz640.csv` e `calibracao/Trotando_2.json` em `analisar_cruzamentos.py`.

Para gerar as figuras em `saidas/figuras/`:

```
python src/gerar_figuras.py
```

## Etapas separadas

A detecção com o YOLO leva alguns minutos por vídeo em CPU. Por isso, as posições de cada quadro já estão em `resultados/keypoints/`. Para refazer:

```
python src/extrair_keypoints.py --video videos/J_Correndo.mp4
```

Para calibrar sem abrir a medição:

```
python src/calibrar.py --video videos/J_Correndo.mp4 --distancia 20
```

## Arquivos

| Arquivo | Função |
|---|---|
| `src/calibrar.py` | Obtém o fator de calibração a partir de duas marcações no primeiro quadro. |
| `src/extrair_keypoints.py` | Aplica o modelo de pose a cada quadro e grava os pontos anatômicos. |
| `src/analisar_cruzamentos.py` | Calcula cruzamentos, velocidades, ruído e a comparação com o cronômetro. |
| `src/kalman.py` | Filtro e suavizador de Kalman e escolha de q. |
| `src/visualizar_medicao.py` | Programa principal: reproduz o vídeo com a medição sobreposta. |
| `src/erros_velocidade.py` | Reúne a comparação com o cronômetro nos três ensaios. |
| `src/caracterizar_kalman.py` | Imprime os parâmetros do filtro em cada ensaio. |
| `src/orcamento_erro.py` | Compara os pontos candidatos e estima as fontes de erro. |
| `src/orcamento_incerteza.py` | Orçamento de incerteza da velocidade média, incerteza da referência e E_n. |
| `src/segmentacao_hsv.py` | Segmentação por cor no Ensaio A. |
| `src/calculo_suavizador.py` | Cálculo do suavizador em um quadro. |
| `src/gerar_figuras.py` | Gera as figuras dos resultados. |
| `src/exibicao.py` | Ajusta os quadros ao tamanho da tela. |
| `calibracao/` | Calibração padrão de cada ensaio. |
| `resultados/cronometragens.csv` | Os 10 tempos de cada ensaio. |
| `resultados/keypoints/` | Pontos anatômicos de cada quadro. |
| `resultados/hsv/` | Posições da segmentação por cor. |
| `videos/` | Vídeos dos ensaios. |
| `yolo26n-pose.pt` | Pesos do modelo de pose. |

Ensaio A: `Trotando_2`, trecho de 5 m. Ensaio B: `J_Correndo`, trecho de 20 m. Ensaio C: `J_Variando`, trecho de 20 m.
