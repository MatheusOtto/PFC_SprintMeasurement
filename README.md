# TCC — Medição de velocidade de corrida por visão computacional

Estima a velocidade de um corredor a partir de vídeo lateral com câmera fixa,
em processamento **offline**, e caracteriza o erro dessa medição. O produto do
trabalho é o software **mais** a caracterização metrológica: um número que melhora
sem explicação é tratado como retrocesso.

O que o sistema mede, com precisão: a velocidade horizontal aparente do **ponto médio
entre os ombros** (keypoints COCO 5 e 6 do YOLO pose), projetada no plano de calibração.

Regras de trabalho, decisões tomadas e problemas conhecidos estão no fim deste arquivo.

---

## Montagem experimental

- Câmera em tripé fixo, vista lateral, sem pan, tilt ou zoom.
- Zona de medição marcada no solo: 5 m nos vídeos antigos; **20 m** com a câmera
  no nível do solo a 21 m da raia (J_Correndo).
- Calibração por dois cliques sobre as marcações → escala única em m/pixel.
- Referência atual: cronômetro manual (até existir a fotocélula).

---

## Pipeline

Rodar sempre a partir da pasta `tcc-velocidade/`, com o ambiente `.venv`
(versões fixadas em `requirements.txt`).

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Calibrar

```
python src/calibrar.py --video videos/J_Correndo.mp4 --distancia 20
```

Abre o primeiro frame numa janela ajustada à tela, com lupa em resolução original
em volta do cursor. Dois cliques nas marcações, `C` confirma. Grava
`calibracao/<video>.json`.

### 2. Extrair os keypoints (roda o YOLO uma vez)

```
python src/extrair_keypoints.py --video videos/J_Correndo.mp4
```

Gera `resultados/keypoints/<video>_imgsz<N>.csv` (uma linha por frame: tempo real e
nominal, número de pessoas, bbox e os 17 keypoints) e um `.json` com versões das
bibliotecas, md5 dos pesos, `imgsz` e limiar de detecção. Usa a **resolução nativa**
do vídeo por padrão; `--imgsz 640` reproduz o comportamento antigo. `--mostrar`
abre uma janela de acompanhamento.

### 3. Analisar

```
python src/analisar_cruzamentos.py resultados/keypoints/J_Correndo_imgsz1920.csv \
    --calibracao calibracao/J_Correndo.json --t-cronometro 3.48
```

Sem rodar o YOLO: detecta os cruzamentos das linhas (interpolação linear), calcula
tempo, velocidade média no trecho, V instantânea (com sinal e com |Δx|), disponibilidade,
ruído dos ombros e compara com o cronômetro. A base de tempo é o carimbo de cada quadro
(desde 2026-10-17); o tempo pelo quadro / FPS do contêiner aparece só como referência. Cruzamento interpolado sobre um intervalo maior
que 0,25 s é marcado **INVÁLIDO** (não descartado). Aceita vários CSVs e imprime uma
tabela lado a lado.

Também roda o **filtro de Kalman** (`src/kalman.py`) sobre a posição dos ombros, com:
- modelo de velocidade quase constante;
- q e R por máxima verossimilhança;
- teste de inovação a 5σ, que rejeita erro de identidade, como o frame 279 do J_Variando;
- suavizador de Kalman com desvio padrão;
- testes de consistência (NIS);
- a velocidade média no trecho com filtro, pela posição suavizada.

O ajuste leva alguns segundos; `--sem-kalman` pula essa parte. Até 2026-10-17 havia também
os modelos CA e com oscilador, comparados por AIC (removidos;
`experimentos/2026-10-07-kalman.md`, Revisão C).

Scripts de apoio, que só imprimem:
- `python src/validar_kalman.py`: valida o filtro com dado sintético de verdade conhecida;
- `python src/caracterizar_kalman.py`: ganho, teste de inovação e resposta em frequência;
- `python src/orcamento_erro.py`: orçamento de erro dos ensaios;
- `python src/sensibilidade_limiar.py`: sensibilidade ao limite do teste de inovação.

### 4. Visualizar

```
python src/visualizar_medicao.py                    # menu para escolher o vídeo
python src/visualizar_medicao.py --video J_Variando
```

Sem argumentos, mostra um menu com os vídeos que já têm keypoints extraídos. O tempo do
cronômetro vem de `resultados/ensaios.csv`. Reproduz o vídeo com linhas, centro dos ombros, lupa no corredor, V instantânea e
gráfico (V bruta em pontos, Kalman com faixa ±2σ, rejeitados em vermelho). No final,
mostra a velocidade média no trecho (distância / tempo) com o tempo do cronômetro e da
visão computacional, sem e com filtro, e a V mínima e a máxima do Kalman suavizado no
trecho. Os números vêm do mesmo `medir()` do analisador. `ESPAÇO` pausa, `+`/`-`
velocidade, `R` recomeça, `S` grava a tela em PNG, `Q` sai.

`--salvar-quadro [N]` não abre janela: grava a tela do quadro N (sem N, a tela final)
em `monografia/figuras/tela_<video>.png`, com 1280 px de largura. É a origem da figura
da ferramenta na monografia.

### Legado (congelado)

`src/Medir_Velocidade_Ombros_Interpolado.py` (medição em um laço só, com interface)
e `src/Testar_Keypoints_YOLO.py` (log dos 17 keypoints) ficam como referência e não
são mais editados.

---

## Estrutura

```
src/                     código (pipeline novo + legado)
calibracao/              calibração por vídeo (JSON)
resultados/ensaios.csv   registro de todos os ensaios, inclusive os ruins
resultados/keypoints/    keypoints brutos (CSV fora do Git; JSON versionado)
diagnostico/             caracterização de cada vídeo (base de tempo, keypoints)
experimentos/            experimentos comparativos: hipótese, métrica, resultado
videos/                  vídeos (fora do Git)
```

---

## Onde estamos

Histórico completo em [`HISTORICO.md`](HISTORICO.md).

| Ensaio | Tempo software | Cronômetro | V média no trecho | Diferença V |
|---|---|---|---|---|
| Trotando_2 (Ensaio A: trote, zona de 5 m) | 2,555 s | 2,55 s | 1,957 m/s | −0,20% |
| J_Correndo (Ensaio B: aceleração) | 3,409 s | 3,48 s | 5,867 m/s | +2,08% |
| J_Variando (Ensaio C: velocidade variada) | 7,237 s | 7,23 s | 2,763 m/s | −0,10% |

Todos gravados com o S24 e analisados com YOLO em resolução nativa, com 100% de cobertura
no trecho.
- J_Correndo e J_Variando: mesma montagem (câmera a 21 m, à altura do tronco, zona de
  20 m), 1920×1080 a 60 fps.
- Trotando_2: primeira gravação, com configuração de vídeo inferior (1280×720 a 30 fps),
  numa rua residencial, com a câmera a 6 m e zona de 5 m entre dois chinelos. Foi
  processado sem mudar nenhum parâmetro.

Com cronômetro manual, as diferenças ainda estão dentro da incerteza da referência.

---

## Pendências

- Sintonia do Kalman (R por verossimilhança, teste a 5σ) confirmada no Trotando_2 sem
  reajuste; falta repetir nos vídeos da sessão com fotocélula. Kalman com q adaptativo
  para surtos dentro da passada.
- Seleção da pessoa por continuidade de trajetória em vez de "maior bbox" (falso
  positivo no frame 279 do J_Variando).
- Analisar sempre o arquivo original da câmera (cópia comprimida falhou).
- Escala no solo vs. altura dos ombros e profundidade da raia (item 3) — decidir
  com o orientador; muda o protocolo de coleta.
- Modelos YOLO maiores (item 5), como experimento comparativo.
- Validar o `GAP_MAX_CRUZAMENTO_S` (0,25 s) com o orientador.
- Mais ensaios e a referência por fotocélula.
- Instalar `ffprobe` para confirmar os carimbos pelo container (a base oficial já é o
  carimbo desde 2026-10-17).

---

## Regras de trabalho

- O produto é o software **mais** a caracterização do erro: um número que melhora sem
  explicação é tratado como retrocesso.
- Nenhuma mudança de método sem comparação antes e depois, com hipótese e critério
  escritos antes de rodar (`experimentos/`).
- Nenhum dado é descartado para melhorar uma estatística; todo ensaio fica em
  `resultados/ensaios.csv`, inclusive os ruins.
- O que é só exibição nunca entra num número do relatório.

## Problemas conhecidos e decisões

Os registros de `diagnostico/` e `experimentos/` citam estes itens pelo número.

1. **Base de tempo.** O tempo pelo quadro dividido pela taxa do contêiner usa uma taxa
   média, distorcida pelo primeiro intervalo nos vídeos a 60 fps. Desde 2026-10-17 a base
   oficial é o carimbo de cada quadro (`CAP_PROP_POS_MSEC`). Falta conferir os carimbos
   pelo contêiner, com `ffprobe`.
2. **`abs(Δx)` enviesa a velocidade instantânea para cima,** porque o módulo retifica o
   ruído. A velocidade instantânea usa o deslocamento com sinal, projetado na direção da
   corrida; a versão com |Δx| fica no relatório só para medir o viés.
3. **Escala única.** A calibração é feita sobre marcas no solo, e o ponto medido fica à
   altura dos ombros e pode passar a outra distância da câmera. A monografia trata a
   montagem como ideal; o efeito está calculado em `diagnostico/orcamento_erro.md`.
4. **Oscilação do tronco.** O centro dos ombros oscila com a passada: é movimento real do
   ponto medido, não erro, mas não é o centro de massa. A velocidade bruta fica no
   relatório sem filtro; ao lado dela, o filtro e o suavizador de Kalman
   (`experimentos/2026-10-07-kalman.md`).
5. **Modelo nano.** `yolo26n-pose` é o menor modelo da família; modelos maiores ficam como
   experimento comparativo. A resolução de entrada já é a nativa do vídeo
   (`experimentos/2026-10-07-imgsz-J_Correndo.md`).
6. **Etapas separadas.** Extração (YOLO) e análise são programas separados, com os
   pontos-chave gravados em CSV: qualquer critério pode ser reavaliado sem nova inferência.
7. **Seleção da pessoa pela maior caixa.** Frágil quando há outras pessoas ou objetos na
   cena (quadro 279 do J_Variando); o teste de inovação do Kalman rejeita o erro. A
   seleção por continuidade de trajetória fica como trabalho futuro.

## Dados

Os vídeos não são públicos, porque mostram pessoas. Os CSV de pontos-chave em
`resultados/keypoints/` permitem reproduzir todos os resultados a partir da etapa de
análise. Os pesos `yolo26n-pose.pt` são baixados pela biblioteca `ultralytics`; o MD5 dos
pesos usados está no JSON ao lado de cada CSV.
