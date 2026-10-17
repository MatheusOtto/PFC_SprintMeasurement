# Diagnóstico — Trotando_2.mp4

Data: 2026-10-17. Arquivo: `videos/Trotando_2.mp4` (cópia de `PFC/Videos/Trotando_2.mp4`,
md5 a8d9642d0cb5f264ca0f542920f334cc). Primeira gravação do trabalho (29/08/2026), com o
Samsung Galaxy S24 em configuração de vídeo inferior, em uma rua residencial.

## 1. Base de tempo

Leitura de todos os quadros com OpenCV, sem inferência (`CAP_PROP_POS_MSEC` após cada
quadro):

| | |
|---|---|
| Resolução | 1280×720 |
| Codec | H.264 |
| Taxa informada pelo contêiner | 30,000 fps |
| Quadros | 179 (6,0 s) |
| Intervalo entre quadros | média 33,333 ms, dp 0,000 ms, mínimo e máximo 33,333 ms |
| Deriva acumulada (carimbo − nominal) no último quadro | +0,0 ms |

**Veredito: CFR**, com uma ressalva. Os intervalos saíram idênticos até a última casa, o
caso em que se deve desconfiar: alguns *backends* do OpenCV derivam
`POS_MSEC` do próprio índice do quadro. Sem `ffprobe` instalado, não foi possível ler os
carimbos do contêiner. No trecho medido, o tempo pelo carimbo e o nominal coincidem
(diferença de 0,00 ms; `analisar_cruzamentos.py`). Desde 2026-10-17 a base oficial é o
carimbo nos três ensaios; neste vídeo as duas bases coincidem.

## 2. Detecção (imgsz 1280, nativo)

- 132 de 179 quadros com pessoa; 123 com os dois ombros com confiança ≥ 0,35.
- No trecho de 5 m: 78 quadros, 78 válidos (100%), maior intervalo 33 ms (1 quadro); nenhum
  quadro com mais de uma pessoa.
- Altura do corredor (caixa) no trecho: mediana 294 px (de 281 a 309 px).
- Ruído do centro dos ombros (segunda diferença): 1,02 px ≈ 0,62 cm.

Os resultados do ensaio e das hipóteses estão em `experimentos/2026-10-17-trotando_2.md`.
