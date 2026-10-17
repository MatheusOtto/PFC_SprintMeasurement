# Histórico do trabalho

Registro cronológico do que foi feito, do que se aprendeu e de onde estamos.
Detalhes técnicos em `diagnostico/`, `experimentos/` e `resultados/ensaios.csv`;
regras e decisões no `README.md`.

---

## Antes deste repositório

1. **Marcadores por cor (HSV)** — rastrear marcadores coloridos no corredor.
   Abandonado.
2. **YOLO pose** — primeiro o centro da bounding box, depois o **centro dos ombros**
   (keypoints COCO 5 e 6), porque a bbox oscila com braços e pernas. Detecção do
   cruzamento das linhas com **interpolação linear** do instante entre as duas
   medições válidas que cercam a linha.
3. **Tentativa de aplicativo completo** (`sprint-measure-app`, Streamlit + SQLite),
   deixada de lado.
4. **`tcc-velocidade`** — recomeço enxuto e focado em metrologia: um arquivo de
   regras (nenhuma mudança de método sem comparação antes/depois, nunca descartar dado
   para melhorar estatística, exibição separada de medição), registro de ensaios e
   experimentos com hipótese declarada antes de medir.

---

## 2026-10-07 — J_Correndo (câmera a 21 m, zona de 20 m)

### Diagnóstico com o script legado
- O `Testar_Keypoints_YOLO.py` (YOLO com `imgsz` 640, o padrão) detectou o corredor em
  só 30% dos frames da passagem. Em 1920×1080, o corredor a 21 m chega ao modelo com
  ~46 px de altura e a confiança fica colada no limiar de detecção (0,25).
- A linha de saída ficava sem medição: o legado interpolaria o cruzamento sobre 3,15 s,
  misturando a corrida com uma pessoa caminhando depois. Resultado: 4,82 m/s,
  **−16% contra o cronômetro**, sem alerta.
- **Reprodutibilidade:** os mesmos pesos no mesmo vídeo detectaram coisas diferentes
  em ultralytics 8.4.133 e 8.4.174. → Versões fixadas no `requirements.txt`.
- **Base de tempo:** 60 fps constantes; o "59,979 fps" do container vem só do primeiro
  intervalo. Efeito no trecho: −2 ms.

### Pipeline novo
O legado (medição + interface no mesmo laço) foi **congelado como referência**. O
pipeline foi separado em etapas com dados brutos salvos:

- `calibrar.py` — dois cliques, janela que cabe na tela (o Windows a 125% cortava a
  janela de 1920 px), lupa em resolução original.
- `extrair_keypoints.py` — roda o YOLO uma vez; grava CSV por frame (tempo real e
  nominal, bbox, 17 keypoints) e JSON com versões e md5 dos pesos.
- `analisar_cruzamentos.py` — cruzamentos, V absoluta, V instantânea, diagnóstico e
  comparação com o cronômetro, sem rodar o YOLO.
- `visualizar_medicao.py` — reproduz o vídeo com a medição sobreposta e gráfico.

Critério novo: cruzamento interpolado sobre intervalo **> 0,25 s é marcado INVÁLIDO**
(não descartado). Justificado por orçamento de erro; a validar com o orientador.

### Experimento `imgsz` (640 × 1280 × 1920)
Hipótese e métrica escritas antes de rodar. Com 1280 e 1920: 100% de disponibilidade
no trecho e cruzamentos entre frames consecutivos; 1920 com menor ruído dos ombros
(0,57 px). **Adotada a resolução nativa do vídeo.** Esta foi a mudança que fez o
J_Correndo funcionar.

### Velocidade instantânea
Implementada em duas versões lado a lado: **com sinal** (projetada na direção da
corrida) e **|Δx|** (como o legado). **Sem filtro:** decisão do autor, a V instantânea
fica bruta e a oscilação da passada é discutida no texto.

No J_Correndo o viés do `abs()` deu **zero**: o ruído (~0,6 px) nunca supera o
deslocamento entre frames (≥ 3,4 px), então nenhum par muda de sinal.

### Resultado (calibração por cliques: x = 263,5 e 1767,5 px)

| | J_Correndo |
|---|---|
| Tempo no trecho | 3,411 s (cronômetro 3,48 s) |
| V absoluta | 5,863 m/s |
| Média das V instantâneas | 5,870 m/s |
| Diferença para o cronômetro | **+2,0%** |

---

## 2026-10-07 — J_Variando (mesma montagem, velocidade variada de propósito)

- **Versão comprimida (848×478):** o primeiro arquivo era uma cópia comprimida.
  Corredor com ~60 px, 32% de detecção, buraco de 1,37 s no trecho → **inválido**.
  Lição de protocolo: **analisar sempre o arquivo original da câmera.**
- **Original (1920×1080):** pessoa em 653/653 frames, 100% de disponibilidade,
  cruzamentos entre frames consecutivos. Mesmo pipeline, sem nenhum parâmetro mudado.
- **Falso positivo no frame 279:** o YOLO viu uma "pessoa" na grade da escada, com
  bbox maior que a do corredor; a regra "maior bbox" escolheu essa detecção por um
  frame. Não afeta a V absoluta; na média com sinal o salto de ida e volta se cancela;
  na média com |Δx| a média vai de 2,77 para **5,12 m/s**. Primeiro caso observado do
  viés do `abs()`, causado por **erro de seleção**, não por ruído.

| | J_Variando |
|---|---|
| Tempo no trecho | 7,228 s (cronômetro 7,23 s) |
| V absoluta | 2,767 m/s |
| Média das V instantâneas (com sinal) | 2,767 m/s |
| V instantânea (bruta) | de 0,66 a 6,70 m/s |
| Diferença para o cronômetro | **+0,03%** |

---

## 2026-10-07 — Filtro de Kalman

Ponto de retorno antes desta etapa: tag `v0-antes-kalman`.

- **Por que:** a V instantânea bruta espalhava de 2,7 a 9,5 m/s no J_Correndo. A
  análise mostrou que a maior parte é ruído do keypoint (~0,6 px) amplificado pela
  derivada a 60 fps (1 px → 0,8 m/s); a oscilação real da passada (3,5 Hz, ~1,3 cm)
  responde por ~0,3 m/s.
- **O que:** `src/kalman.py` — filtro com Δt variável (frame sem medição vira só
  predição), teste de inovação, suavizador RTS, sintonia por máxima verossimilhança
  e três modelos comparados por AIC (CV, CA e CV + oscilador harmônico da passada).
  Validado antes com dado sintético (`src/validar_kalman.py`).
- **Ajustes depois de ver os dados (declarados no experimento):** R por máxima
  verossimilhança (saiu 1,26 × σx nos dois vídeos) e teste de inovação a 5σ — com 3σ
  o filtro descartava um surto real de velocidade dentro da passada.
- **Resultado** (`experimentos/2026-10-07-kalman.md`): dispersão da V instantânea
  ~3,5× menor (J_Correndo 0,91 → 0,26 m/s; J_Variando 0,66 → 0,24 m/s, em torno da
  tendência); V absoluta muda < 0,1%; o frame 279 é rejeitado sozinho, sem descartar
  nenhuma outra medição; NIS ≈ 1. **O modelo CV venceu no AIC nos dois vídeos**: a
  passada existe e é encontrada pelo modelo com oscilador, mas é pequena demais frente
  ao ruído para justificar os parâmetros a mais.
- **Visualizador:** a média móvel de exibição saiu; o gráfico mostra a curva do
  Kalman com faixa ±2σ (o intervalo de confiança vindo da covariância).

| | V abs interpolação | V abs Kalman | Cronômetro |
|---|---|---|---|
| J_Correndo | 5,8634 m/s | 5,8576 m/s | 5,7471 m/s |
| J_Variando | 2,7670 m/s | 2,7687 m/s | 2,7663 m/s |

---

## 2026-10-08 — Orçamento de erro e início da monografia em LaTeX

- **Orçamento de erro** (`src/orcamento_erro.py`, `diagnostico/orcamento_erro.md`): o
  ruído do keypoint desloca o tempo de cada cruzamento em 1,8 a 3,7 ms, o que dá 0,06% a
  0,11% na V absoluta; com interpolação, a taxa de quadros pouco importa (10 fps com
  interpolação erra menos que 30 fps sem). Os erros maiores são sistemáticos: ponto do
  corpo (0,7% a 2,4%) e profundidade (0,48% a cada 10 cm, calculado). É precisão, não
  exatidão: a referência ainda é o cronômetro.
- **Monografia** em `monografia/` (branch `monografia`), a partir do rascunho abnTeX2 do
  autor, com títulos e ordem preservados. Raspberry Pi e tempo real saem do texto.
  Capítulo 3 reescrito, Capítulo 4 (Resultados) novo, figuras por `src/gerar_figuras.py`.
  Compilar com `monografia/compilar.ps1` (MiKTeX).
- **Referências:** o `.bib` foi conferido contra o Crossref; havia uma entrada que
  misturava o título do DeepSORT com os dados do SORT e páginas cortadas.

---

## 2026-10-17 — Trotando_2, caracterização do filtro e revisão da monografia

- **Trotando_2** (`experimentos/2026-10-17-trotando_2.md`): primeira gravação (29/08), em
  1280×720 a 30 fps, zona de 5 m entre chinelos, cronômetro de 2,55 s.
  - Rodado no pipeline atual **sem mudar nenhum parâmetro**, com hipóteses escritas antes.
  - Trecho medido a 100%, cruzamentos entre quadros consecutivos, V absoluta 1,9569 m/s
    (−0,20% do cronômetro).
  - As duas escolhas do Kalman feitas depois de ver os dados se mantiveram: κ = 1,26, NIS
    1,03, nenhuma rejeição.
  - **Hipótese refutada:** o centro do quadril teve menos ruído que o dos ombros (0,93 ×
    1,02 px). Nos dois sprints a 60 fps, os ombros tiveram o menor ruído (0,58 e 0,42 px,
    contra 0,81 e 0,58 px do quadril).
- **Caracterização do filtro** (`src/caracterizar_kalman.py`,
  `diagnostico/kalman_caracterizacao.md`):
  - ganho de Kalman em regime de 0,54 a 0,60 na posição;
  - o maior desvio de uma medição boa foi 4,2σ, e o erro de identidade do quadro 279 deu
    758σ, o que justifica o limite de 5σ;
  - o suavizador equivale a um passa-baixas sem atraso, com corte de 4,7 a 5,1 Hz nos
    sprints e de 2,2 Hz no trote.
- **Orçamento de erro** generalizado para os três ensaios. O ruído e a disponibilidade por
  ponto do corpo agora ficam registrados.
- **Figuras:** legenda "Kalman suavizado"; figura nova do Trotando_2.
- **Monografia reescrita:**
  - saem AIC, Bland–Altman, o nome RTS, Thorpe, Gabbett e os pontos físicos da montagem;
  - entram o GPS no estado da arte, todas as contas do filtro e a escolha do ponto com
    dados;
  - referências fechadas, todas lidas: Hopkins (2000), Kalman (1960), Särkkä e Svensson
    (2023), YOLO26 (2026) e a documentação da Ultralytics.
  - Cópia anterior em `Codigo/backup_2026-10-17_antes-revisao/`.
  - **Citações enxugadas:** cada premissa é citada uma vez e não é retomada por "(Seção
    X)". O texto passou de 85 para 25 citações, com nenhum artigo mais de 2 vezes por
    capítulo; nenhum número mudou. A regra está em `monografia/pendencias.md`, seção 4.
- **Código do Kalman: Revisão C** (`experimentos/2026-10-07-kalman.md`, seção 7):
  - **Só o modelo CV.** Saíram o CA, o oscilador e o AIC. Antes da remoção, a comparação
    foi refeita sem o viés de contagem: o CA perde nos três ensaios; o oscilador ganha
    pouco em dois e perde em um, e nunca supera o CV pelo BIC.
  - **Verificação:** o relatório do CV ficou idêntico depois da remoção.
  - **Primeira medição:** passou a só iniciar o estado (antes era usada duas vezes). O
    efeito ficou na 3ª casa da autocorrelação e do NIS, nos ganhos das primeiras
    medições e no ±σ dos testes de taxa reduzida. Nenhum número do texto mudou além
    desses.
  - **`validar_kalman.py` reescrito para o CV:** aprovado nos 4 critérios (NIS 0,98,
    cobertura ±2σ de 100%, RMS 0,086 m/s).
  - **`visualizar_medicao.py`:** só a curva do Kalman suavizado, e a distância do trecho
    lida da calibração (antes fixa em 20 m).
- **Tela de resultado enxuta e figura da ferramenta:**
  - O painel final mostra a V absoluta (distância / tempo) com o tempo de três origens:
    cronômetro, visão computacional sem Kalman e com Kalman. Saíram as médias das
    velocidades instantâneas, redundantes com a V absoluta.
  - Entraram a V mínima e a máxima do Kalman suavizado no trecho, calculadas em
    `aplicar_kalman` e impressas no relatório. As linhas antigas do relatório ficaram
    idênticas nos três ensaios.
  - O texto da tela passou a ter acentos (Pillow), e a escala do gráfico acompanha a V
    máxima.
  - A tela é gravada pelo próprio script com `--salvar-quadro` ou com a tecla S. A tela
    final do J_Variando entrou na monografia como figura do Cap. 3.
- **Base de tempo pelo carimbo** (revisão do PDF de 17/10, item 5.1):
  - O tempo oficial era o quadro dividido pela taxa do contêiner (59,979 e 60,077 fps nos
    vídeos de 60 fps), uma média distorcida pelo 1º intervalo anômalo. Passou a ser o
    carimbo de cada quadro (`dados["t"]`, `carregar_keypoints`); o nominal fica no
    relatório como referência.
  - Efeito: J_Correndo 5,8634 → 5,8669 m/s (+2,02% → +2,08% contra o cronômetro);
    J_Variando 2,7670 → 2,7634 m/s (+0,03% → −0,10%); Trotando_2 igual. Nenhuma
    conclusão muda. Relatórios antes e depois comparados linha a linha: só mudou a escala
    do tempo.
  - Refeitos: tabela de sensibilidade ao limite (agora em `src/sensibilidade_limiar.py`),
    caracterização, orçamento de erro, figuras e telas.
- **Revisão da monografia pelo PDF de 17/10:**
  - "velocidade absoluta" virou "velocidade média no trecho", dada de três formas
    (cronômetro, visão computacional sem e com filtro), no texto, nas figuras e na tela;
  - "repetibilidade" virou "contribuição do ruído de posição para a incerteza", com o VIM
    citado uma vez;
  - ensaios renomeados para A (Trotando_2), B (J_Correndo) e C (J_Variando);
  - resultados tirados do Cap. 3; teste sintético com subseção, figura
    (`sintetico_kalman.pdf`) e tabela; Tabela de efeito do filtro; uma equação só para a
    velocidade média e outra para o fator C;
  - calibração com a resolução de clique do JSON (1,48 px nos vídeos de 1920 px: 0,07%, e
    não 0,05%);
  - Figura 1 sem sobreposições, `
aggedbottom`, figuras grandes flutuantes, Apêndice A
    (repositório e programa de cada tabela e figura).

---

## Onde estamos

**Três ensaios válidos** com o pipeline novo, em duas montagens:

| Ensaio | Tempo software | Cronômetro | Diferença V |
|---|---|---|---|
| Trotando_2 (5 m, 30 fps) | 2,555 s | 2,55 s | −0,20% |
| J_Correndo (20 m, 60 fps) | 3,409 s | 3,48 s | +2,08% |
| J_Variando (20 m, 60 fps) | 7,237 s | 7,23 s | −0,10% |

Com três ensaios e cronômetro manual, **ainda não dá para afirmar viés** em nenhum
sentido. Haugen e Buchheit (2016) relatam 0,19 ± 0,14 s entre cronometristas, e as três
diferenças estão dentro dessa incerteza. O que já está demonstrado é a parte técnica:
- com o arquivo original e a resolução nativa, o trecho é coberto a 100%;
- a velocidade média no trecho e a média das instantâneas concordam em ~0,1%;
- o Kalman se mantém em um vídeo novo sem reajuste.

### Próximos passos
- Sintonia do Kalman confirmada no Trotando_2 sem reajuste; repetir nos vídeos da sessão
  com fotocélulas. Testar q adaptativo.
- **Seleção da pessoa** por continuidade de trajetória em vez de "maior bbox"
  (experimento comparativo nos dois vídeos).
- Mais ensaios; referência por fotocélula.
- Escala no solo vs. altura dos ombros e profundidade da raia (problema conhecido 3 do README)
  — decidir com o orientador; muda o protocolo de coleta.
- Modelos YOLO maiores (item 5), como experimento.
- Validar o `GAP_MAX_CRUZAMENTO_S` (0,25 s) com o orientador.
- Instalar `ffprobe` para confirmar os carimbos pelo container.
