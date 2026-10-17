# Caracterização do filtro de Kalman (modelo CV)

Data: 2026-10-17. Script: `src/caracterizar_kalman.py` (`python src/caracterizar_kalman.py`,
só imprime). Números atualizados depois da Revisão C (`experimentos/2026-10-07-kalman.md`,
seção 7): a primeira medição da janela agora só inicia o estado. Atualizado de novo em
2026-10-17, com a base de tempo pelo carimbo (60,00 fps nos dois vídeos de 20 m; antes,
59,98 e 60,08 pela taxa do contêiner). Mudaram só a 3ª casa de alguns ganhos, o √S do
J_Variando e o 758σ, que virou 757σ. Para cada ensaio: CSV de keypoints de maior imgsz, q e R por máxima
verossimilhança no próprio ensaio (como em `analisar_cruzamentos.py`), janela de 0,5 s antes
da entrada a 0,5 s depois da saída.

## 1. Parâmetros e o que significam

| | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| Taxa de quadros | 30,00 fps | 60,00 fps | 60,00 fps |
| Escala (C) | 0,6068 cm/px | 1,3298 cm/px | 1,3298 cm/px |
| q | 0,151 m²/s³ | 2,754 m²/s³ | 1,995 m²/s³ |
| Variação típica da velocidade em um quadro, √(q·Δt) | 0,071 m/s | 0,214 m/s | 0,182 m/s |
| Variação típica da velocidade em 1 s, √q | 0,39 m/s | 1,66 m/s | 1,41 m/s |
| σx (segunda diferença, MAD) | 1,023 px | 0,576 px | 0,422 px |
| √R | 0,781 cm = 1,287 px | 0,964 cm = 0,725 px | 0,706 cm = 0,531 px |
| κ = √R / (σx·C) | 1,2589 | 1,2589 | 1,2589 |

O κ é o mesmo ponto da grade nos três ensaios (10^0,10; a grade tem passos de 10^0,05,
isto é, 12%).

## 2. Ganho de Kalman, K = P⁻·Hᵀ / S

[posição (adimensional); velocidade (1/s)], por ordem da medição na janela:

| Medição | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| 1ª | inicia o estado (sem ganho) | inicia o estado (sem ganho) | inicia o estado (sem ganho) |
| 2ª | [0,999; 29,97] | [0,997; 59,61] | [0,998; 59,79] |
| 3ª | [0,835; 15,22] | [0,835; 30,62] | [0,836; 30,88] |
| 5ª | [0,627; 7,41] | [0,638; 16,05] | [0,650; 17,25] |
| 10ª | [0,541; 6,17] | [0,578; 14,48] | [0,605; 16,26] |
| **Regime** (mediana a partir da 20ª) | **[0,541; 6,16]** | **[0,577; 14,46]** | **[0,605; 16,24]** |
| β = K_vel·Δt | 0,205 | 0,241 | 0,270 |

Leitura do ganho em regime:
- **Trotando_2:** a posição estimada anda 54% em direção a cada medida, e 1 cm de inovação
  muda a velocidade em 6,2 cm/s.
- **J_Correndo:** 58%, e 1 cm muda a velocidade em 14,5 cm/s.
- **J_Variando:** 60%, e 1 cm muda a velocidade em 16,2 cm/s.

Por que o ganho começa assim:
- **1ª medição, sem ganho.** Ela só inicia o estado: posição igual à medição, com variância
  R. Até a Revisão C, ela era usada também na primeira atualização (ganho 0,5).
- **2ª medição, ganho de posição ≈ 1 e ganho de velocidade ≈ fps.** A velocidade inicial é
  desconhecida (desvio de 10 m/s), então a segunda medição é aceita quase inteira e a
  velocidade vira a diferença entre as duas primeiras medições.
- **A partir da 10ª medição** o ganho já está no valor de regime.

## 3. Teste de inovação (limiar NIS > 25, isto é, 5σ)

| | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| Medições na janela (a 1ª inicia o estado) | 107 | 265 | 495 |
| √S mediano (desvio esperado da inovação) | 1,153 cm | 1,482 cm | 1,123 cm |
| Rejeitados | nenhum | nenhum | quadro 279: inovação −8,503 m, √S 1,123 cm → **757σ** (NIS 5,74·10⁵) |
| Maior inovação aceita | 2,90σ (quadro 70, +3,35 cm) | 3,89σ (quadro 387, −5,76 cm) | **4,17σ** (quadro 317, +4,69 cm; surto real de velocidade) |
| Aceitas acima de 3σ / 4σ | 0 / 0 | 4 / 0 | 8 / 1 |

Chance de uma medição boa passar do limiar se o erro fosse gaussiano,
P(|ν| > nσ) = erfc(n/√2), e rejeições esperadas nas 493 medições aceitas que passam pelo
teste no J_Variando:

| Limiar | Probabilidade | Esperado em 493 |
|---|---|---|
| 3σ (NIS > 9) | 2,700·10⁻³ | 1,33 |
| 4σ (NIS > 16) | 6,334·10⁻⁵ | 0,031 |
| **5σ (NIS > 25)** | **5,733·10⁻⁷** | **0,0003** |
| 10σ (NIS > 100) | 1,524·10⁻²³ | 0 |

Leitura: qualquer limiar entre o maior desvio de uma medição boa (4,17σ) e o erro de
identidade (757σ) separa os dois. 5σ fica acima do primeiro com folga. A 3σ, que seria o
usual, o filtro rejeitou 9 quadros no J_Variando (`experimentos/2026-10-07-kalman.md`,
seção 4.1) contra 1,33 esperados para erro gaussiano: o erro real tem cauda mais pesada, e
5 desses quadros eram um surto real de velocidade.

## 4. Filtro causal × suavizador

Desvio padrão médio da velocidade estimada no trecho:

| | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| Filtro causal | 0,104 m/s | 0,295 m/s | 0,240 m/s |
| Suavizador | 0,057 m/s | 0,163 m/s | 0,134 m/s |

## 5. Consistência (medições aceitas, sem as 2 da inicialização)

| | Trotando_2 | J_Correndo | J_Variando |
|---|---|---|---|
| NIS médio (esperado 1) | 1,026 | 1,015 | 1,003 |
| Fração acima de 3,84 (esperado 5%) | 4,8% | 6,5% | 5,1% |
| Autocorrelação lag-1 das inovações normalizadas | +0,215 | +0,080 | +0,112 |
| Tolerância ±2/√n | ±0,195 (n = 105) | ±0,123 (n = 263) | ±0,090 (n = 492) |

A autocorrelação passa um pouco da tolerância no Trotando_2 e no J_Variando. A oscilação da
passada, que o modelo de velocidade quase constante não representa, deixa parte da
inovação de um quadro parecida com a do quadro seguinte.

Ressalva: q e R são escolhidos nestes mesmos dados, por máxima verossimilhança. Por isso
o NIS médio perto de 1 é, em parte, consequência do ajuste. As verificações independentes
são o dado sintético (`src/validar_kalman.py`) e o Trotando_2, que não participou da escolha
do limiar nem do fator de R.

## 6. Resposta em frequência na velocidade

Senoide de posição de 1 cm somada a uma progressão constante de 3 m/s, sem ruído, 12 s,
filtrada com o q e o R de cada ensaio. Ganho = amplitude estimada / amplitude verdadeira
da velocidade; atraso pela fase; descartados 3 s em cada ponta.

| f (Hz) | Trotando_2: filtro (atraso) | Trotando_2: suavizador | J_Correndo: filtro (atraso) | J_Correndo: suavizador | J_Variando: filtro (atraso) | J_Variando: suavizador |
|---|---|---|---|---|---|---|
| 0,5 | 1,000 (72 ms) | 0,999 | 1,000 (32 ms) | 1,000 | 0,999 (29 ms) | 0,999 |
| 1 | 0,991 (75 ms) | 0,979 | 1,000 (32 ms) | 0,999 | 0,999 (29 ms) | 0,998 |
| 2 | 0,872 (81 ms) | 0,749 | 0,995 (33 ms) | 0,987 | 0,996 (30 ms) | 0,988 |
| 3,5 | 0,502 (75 ms) | 0,241 | 0,945 (35 ms) | 0,884 | 0,960 (32 ms) | 0,910 |
| 5 | 0,279 (60 ms) | 0,071 | 0,813 (36 ms) | 0,646 | 0,854 (33 ms) | 0,710 |
| 7 | 0,153 (44 ms) | 0,019 | 0,580 (35 ms) | 0,322 | 0,640 (32 ms) | 0,390 |
| 10 | 0,083 (30 ms) | 0,004 | 0,335 (29 ms) | 0,102 | 0,383 (28 ms) | 0,133 |
| 15 | — | — | 0,165 (20 ms) | 0,021 | 0,191 (20 ms) | 0,029 |
| 20 | — | — | 0,102 (15 ms) | 0,006 | 0,118 (14 ms) | 0,008 |
| **Corte (ganho 0,707)** | **2,7 Hz** | **2,2 Hz** | **5,9 Hz** | **4,7 Hz** | **6,4 Hz** | **5,1 Hz** |

O suavizador não tem atraso, porque a fase é zero. O filtro causal atrasa de 29 a 36 ms nos
ensaios a 60 fps e de 30 a 81 ms no Trotando_2.

Leitura: o filtro equivale a um passa-baixas cuja frequência de corte é definida pelos
dados, pela razão entre q e R.
- **Sprints (J_Correndo, J_Variando):** o suavizador corta perto de 5 Hz. Mantém 88% a 91%
  da oscilação da passada a 3,5 Hz e remove mais de 97% do que está acima de 15 Hz.
- **Trote em ritmo constante (Trotando_2):** a verossimilhança escolheu q pequeno, e o
  corte cai para 2,2 Hz. A oscilação a 3,5 Hz fica com 24%.
