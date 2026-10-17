# Orçamento de erro — Trotando_2, J_Correndo e J_Variando

Primeira versão: 2026-10-08 (J_Correndo e J_Variando). Atualizado em 2026-10-17: Trotando_2
incluído, e ruído e disponibilidade de cada ponto do corpo registrados. Atualizado de novo em
2026-10-17: a base de tempo oficial passou a ser o carimbo de cada quadro, e a calibração
passou a usar a resolução de clique gravada no JSON (os números abaixo já estão na base
nova).

- **Script:** `src/orcamento_erro.py` (`python src/orcamento_erro.py`, só imprime).
- **Dados:** para cada ensaio, o CSV de keypoints de maior imgsz (nativo) e
  `calibracao/*.json`.
- **Pipeline:** `analisar_cruzamentos.medir()` com Kalman (modelo CV).
- **Montagens:**
  - J_Correndo e J_Variando: 1920×1080 a 60 fps, zona de 20 m, câmera a 21 m;
  - Trotando_2: 1280×720 a 30 fps, zona de 5 m.

Mesmo corredor nos três.

Pergunta: o erro do tempo de cruzamento, que é o que a taxa de quadros limita, acompanha a
precisão que o uso exige? E o que domina o erro total?

## 1. Resultado resumido

Efeito na velocidade absoluta (Trotando_2 / J_Correndo / J_Variando):

| Fonte de erro | Efeito | Situação |
|---|---|---|
| Ruído do keypoint no tempo dos cruzamentos (com interpolação) | 0,23% / 0,11% / 0,06% | medido |
| Só a quantização do quadro, sem interpolar | desvio 0,53% / 0,20% / 0,09%; pior caso 1,30% / 0,49% / 0,23% | medido |
| Base de tempo (resíduo dos carimbos: desvio dos intervalos de 0 / 0,019 / 0 ms) | < 0,001% nos três | medido; falta conferir com `ffprobe` |
| Calibração por cliques (meia resolução de clique em cada linha: 0,5 px no Trotando_2; 0,74 px nos vídeos de 1920 px, cuja janela de calibração é reduzida para caber na tela) | ≈ 0,09% (sobre 824 px) / ≈ 0,07% / ≈ 0,07% (sobre 1504 px) | estimado |
| Ponto do corpo usado | 0,07% a 4,6% / 0,7% a 2,4% / 0,1% a 1,5% | medido (seção 3) |
| Troca de pessoa (quadro 279 do J_Variando) | erro grosseiro | o Kalman rejeita; a seleção pela maior caixa continua frágil |

**Fora do texto da monografia** (decisão do autor, 2026-10-17: a montagem física é tratada
como perfeita). Ficam registrados aqui:

| Item | Valor |
|---|---|
| *Rolling shutter* | < 0,02% / < 0,005% / < 0,005% (ombros quase à mesma altura nas duas linhas) |
| Profundidade (corredor fora do plano das marcas) | 0,48% a cada 10 cm, só na montagem de 21 m; calculado, não medido |
| Distorção da lente | não medida |

**Correção da base de tempo (2026-10-17).** Até esta data, a base oficial era o quadro
dividido pela taxa do contêiner (59,979 e 60,077 fps nos vídeos de 60 fps). Trocar pelo
carimbo mudou a velocidade média em +0,06% no J_Correndo e −0,13% no J_Variando; o
Trotando_2 não mudou. A linha "Base de tempo" acima passou a ser o resíduo dos carimbos.

## 2. Efeito da taxa de quadros

**Tempo de cada linha.** σt = σz / v, com σz o desvio do ruído de medição estimado pelo
Kalman e v a velocidade do corredor na linha:

| Ensaio | σz | σt por linha | σ(V)/V |
|---|---|---|---|
| Trotando_2 | 1,29 px = 7,8 mm | 4,1 a 4,2 ms | 0,23% |
| J_Correndo | 0,72 px | 1,8 a 3,4 ms | 0,11% |
| J_Variando | 0,53 px | 2,6 a 3,7 ms | 0,06% |

O σt fica de 4,5 a 9,5 vezes abaixo do intervalo entre quadros (33,3 ms a 30 fps; 16,7 ms a
60 fps).

**Decimação.** Quadros removidos, em todas as fases possíveis. Maior desvio da V absoluta em
relação ao resultado na taxa original:

| Ensaio (taxa original) | Taxa | Com interpolação | Sem interpolação |
|---|---|---|---|
| J_Correndo / J_Variando (60 fps) | 30 fps | 0,08% / 0,07% | 0,69% / 0,40% |
| | 20 fps | 0,15% / 0,10% | 1,17% / 0,17% |
| | 15 fps | 0,26% / 0,08% | 1,64% / 0,52% |
| | 10 fps | 0,31% / 0,10% | 2,58% / 0,86% |
| Trotando_2 (30 fps) | 15 fps | 0,01% | 1,73% |
| | 10 fps | 0,11% | 2,20% |
| | 7,5 fps | 0,15% | 4,19% |
| | 5 fps | 0,47% | 6,46% |

**Velocidade em janela de duração T** (regressão linear; erro padrão mediano, m/s):

| T | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| 1 quadro (bruta, teórico) | 0,33 | 0,82 | 0,60 |
| 0,10 s | 0,09 | 0,11 | 0,08 |
| 0,286 s (um ciclo de 3,5 Hz, a cadência medida no J_Correndo) | 0,04 | 0,04 | 0,03 |

**±σ mediano do Kalman suavizado** na taxa original e decimada:

| Ensaio | Taxa original | Metade da taxa | Um quarto da taxa |
|---|---|---|---|
| Trotando_2 (30 / 15 / 7,5 fps) | 0,057 | 0,052 | 0,048 m/s |
| J_Correndo (60 / 30 / 15 fps) | 0,163 | 0,170 | 0,196 m/s |
| J_Variando (60 / 30 / 15 fps) | 0,134 | 0,139 | 0,148 m/s |

## 3. Ponto do corpo

Mesmas linhas e mesmo critério de confiança (0,35 em cada ponto usado). Disponibilidade no
trecho de 100% para todos os pontos nos três ensaios.

| Ponto | σx (px): Trotando_2 / J_Correndo / J_Variando | V absoluta contra o centro dos ombros: Trotando_2 / J_Correndo / J_Variando |
|---|---|---|
| centro dos ombros (5, 6) | **1,02 / 0,58 / 0,42** | — |
| ombro esquerdo (5) | 1,02 / 0,95 / 0,56 | +4,08% / +0,74% / +1,44% |
| ombro direito (6) | 1,78 / 1,50 / 0,89 | −4,57% / −1,08% / −1,45% |
| centro do quadril (11, 12) | 0,93 / 0,81 / 0,58 | +0,07% / +1,97% / +0,72% |
| nariz (0) | 0,63 / 0,59 / 0,58 | +1,26% / −1,11% / +0,11% |
| centro da caixa delimitadora | 1,92 / 1,34 / 0,68 | +1,51% / +2,44% / +1,53% |

Leitura:
- **Sprints a 60 fps:** o centro dos ombros teve o menor ruído. No J_Correndo, o nariz
  empatou (0,59 px). O centro do quadril teve 40% e 38% a mais.
- **Trote a 30 fps (Trotando_2):** o quadril teve 9% menos ruído que os ombros, e o nariz
  foi o menor (0,63 px). Hipótese **não testada:** com quadros 67 ms afastados, a segunda
  diferença capta mais a oscilação horizontal real dos ombros (rotação do tronco, braços).
- **Cada ombro isolado** dá velocidade de 0,7% a 4,6% diferente do centro, em sentidos
  opostos nos dois ombros, e o centro cancela esse efeito.
- **Caixa delimitadora:** de 1,6 a 2,3 vezes o ruído dos ombros nos sprints e 1,9 vez no
  trote.

## 4. Ressalvas

- Três ensaios, um corredor. Nada aqui é estimativa da variação entre atletas, dias ou
  câmeras.
- A decimação remove quadros do mesmo vídeo. É o efeito da amostragem sobre o tempo de
  cruzamento, não uma previsão para outra câmera.
- Para T ≥ 0,5 s o erro padrão empírico da janela fica acima do teórico de ruído: a
  regressão linear passa a medir também a aceleração real do corredor.
- A referência dos três ensaios é o cronômetro manual. Haugen e Buchheit (2016) relatam
  0,19 ± 0,14 s entre cronometristas na mesma tentativa, o que dá 7,5%, 5,5% e 2,6% destes
  tempos. Ele não permite concluir se o sistema tem viés. Este orçamento descreve a
  **precisão** (repetibilidade do cálculo), não a **exatidão**.

## 5. Pendências que este orçamento aponta

1. Referência por fotocélulas: erro típico contra a fotocélula.
2. Base de tempo pelo contêiner (`ffprobe`).
3. Ponto de referência: ombros, quadril ou nariz. Com três ensaios, os ombros têm o menor
   ruído em dois. Repetir a comparação nos vídeos da sessão com fotocélulas.
4. Seleção da pessoa por continuidade de trajetória.
