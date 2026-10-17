# Diagnóstico — J_Correndo.mp4

Data: 2026-10-07
Arquivo: `videos/J_Correndo.mp4`, 1920×1080, 807 frames (~13,4 s), gravado como 60 fps.
Montagem: câmera no nível do solo, do outro lado da quadra, a 21 m da linha de
corrida; zona de **20 m** marcada por duas garrafas brancas na borda da grama.
Corrida da direita para a esquerda. O corredor largou parado ao lado da garrafa da
direita (informação do autor) e acelera ao longo do trecho; nas imagens ele aparece
na borda direita do quadro, à direita da linha de entrada, até ~frame 290.
Cronômetro manual: **3,48 s**.

---

## 1. Base temporal

Sem `ffprobe` instalado; carimbos lidos com `CAP_PROP_POS_MSEC` após cada `read()`
(o primeiro frame retorna 0,0 ms, ou seja, o valor é o do frame recém-lido). Os
intervalos **não** são todos idênticos (variam em passos de ~0,011 ms), então o teste
não é circular — mas a confirmação pelo container com `ffprobe -show_frames` continua
pendente.

| Grandeza | Valor |
|---|---|
| FPS informado pelo container | 59,9789 (16,6725 ms) |
| Mediana dos intervalos | **16,6667 ms (60,000 fps)** |
| Média / desvio, todos os intervalos | 16,6725 / 0,278 ms |
| Média / desvio, sem o 1º intervalo | 16,6628 / **0,019 ms** |
| Mínimo / máximo, sem o 1º intervalo | 16,467 / 16,911 ms |
| Intervalos a mais de 20% da mediana | 1 — frame 0→1, **24,544 ms** |
| Deriva carimbo − nominal | +7,87 ms no frame 1, diluindo até +0,006 ms no último |
| Efeito no trecho de 20 m (frames ~291–496) | **−2,0 ms** em ~3,4 s (0,06%) |

**Veredito: CFR confiável.** A câmera grava a 60 fps de fato; o "59,979" do container é
artefato do primeiro intervalo (24,5 ms), que o container dilui na média. Usar o FPS
nominal neste vídeo erra o tempo do trecho em ~2 ms — desprezível frente à incerteza do
cronômetro manual, e registrado aqui como resultado. O pipeline novo grava as duas bases
(`t_ms` e `t_nominal_s`) e o analisador reporta ambas.

**Atualização 2026-10-17:** a base oficial passou a ser o carimbo
(`analisar_cruzamentos.carregar_keypoints`). O quadro / FPS do contêiner usa a média de
59,979 fps, distorcida pelo 1º intervalo; os carimbos sem esse intervalo dão 16,6628 ms
(60,014 fps), o que explica exatamente os −2,0 ms. Com a base nova: 3,4090 s e 5,8669 m/s
(+2,08% contra o cronômetro), antes 3,4110 s e 5,8634 m/s (+2,02%).

---

## 2. Keypoints com o setup legado (`imgsz` 640, yolo26n-pose)

Backlog: `resultados/backlogs_antigos/J_Correndo_07-10/` (mesmo conteúdo que
`resultados/keypoints/J_Correndo_imgsz640.csv`; diferença máxima 6·10⁻⁵ px).

- **Gargalo é a detecção da pessoa, não os keypoints.** Pessoa detectada em 134/807
  frames (16,6%); durante a passagem (frames 259–479) em 30%. Quando há detecção, os dois
  ombros passam no limiar 0,35 em 91% dos frames (conf. mediana 0,97 / 0,94).
- O corredor tem ~137 px de altura no vídeo e chega ao modelo com **~46 px** (640/1920).
  A confiança da pessoa fica entre 0,25 e 0,59 (mediana 0,35) — colada no limiar padrão
  de detecção (0,25), por isso a detecção liga e desliga.
- Correndo: 30% de detecção; caminhando depois da corrida: 68%. Movimento rápido piora
  (hipótese: borrão de movimento).
- Detecções em rajadas a cada ~15 frames (250 ms, ~4 Hz), ritmo compatível com a passada.
  **Hipótese** não verificada: se a detecção depende da fase da passada, o descarte não é
  aleatório.
- Ruído de x do centro dos ombros correndo: ~1,4 px (2ª diferença, n pequeno).
  O ombro do lado oculto é mais ruidoso que o visível (1,96 vs 1,23 px), mesmo padrão do
  Trotando_2 (2,87 vs 1,78 px).
- Falsos positivos: frames 140/142 na borda direita (caixa fina, conf 0,25); frames
  658–754 são uma pessoa caminhando na borda esquerda depois da corrida (provavelmente o
  próprio corredor voltando).

**Consequência para a medição:** a linha de saída (x = 263) fica sem medição válida
próxima. O pipeline interpolaria a saída entre o frame 479 (x = 382, ainda correndo) e o
668 (x = 12, já a pessoa caminhando de volta): gap de 189 frames (3,15 s), misturando dois
movimentos. Resultado: 4,150 s → 4,82 m/s, **−16,2% contra o cronômetro**, sem nenhum
alerta no legado além do campo `gap_saida = 189`.

---

## 3. Reprodutibilidade entre ambientes

Mesmos pesos (md5 `af743a72087d8101c1025b9165056f79`), mesmo script, mesmo vídeo:

| Ambiente | Frames com pessoa |
|---|---|
| Python global, ultralytics 8.4.133 / torch 2.13.0 (gerou o backlog de 17-09) | 89 |
| `.venv`, ultralytics 8.4.174 / torch 2.14.1 | 134 |

Confirmado reexecutando os dois ambientes nos frames 270–282: o 8.4.133 reproduz 17-09
exatamente. O 8.4.174 dá confianças de pessoa 0,03–0,10 maiores; como elas estão em cima
do limiar 0,25, uma atualização de biblioteca muda quais frames entram. Nos frames
detectados pelas duas versões, os keypoints diferem 1–2 px. (O Anaconda, ultralytics
8.3.233, nem carrega o yolo26.)

Medidas tomadas: versões fixadas no `requirements.txt`; cada extração grava as versões e
o md5 dos pesos no JSON ao lado do CSV.

---

## 4. Calibração

Confirmada por cliques em `src/calibrar.py` (`calibracao/J_Correndo.json`):
x = 263,5 e 1767,5 px → 1504 px = 20 m → 1,330 cm/px (resolução de clique 1,48 px).
A estimativa provisória anterior, por inspeção ampliada do frame 0 + centroide de pixels
claros, era 263 e 1766 px (1503 px): diferença de 0,07% na escala. Coerente com a altura
do corredor (bbox ~137 px ≈ 1,8 m).

---

## 5. Efeito do `imgsz`

Ver `experimentos/2026-10-07-imgsz-J_Correndo.md`.
