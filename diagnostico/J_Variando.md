# Diagnóstico — J_Variando.mp4

Data: 2026-10-07
Arquivo: `videos/J_Variando.mp4`, 1920×1080, 653 frames (~10,9 s), 60 fps.
Montagem: a mesma do J_Correndo (câmera no nível do solo a 21 m da raia, zona de
20 m entre as garrafas, tripé no mesmo lugar). Corrida da direita para a esquerda com
**velocidade variada de propósito**. Cronômetro manual: **7,23 s**.

---

## 1. Base temporal

| Grandeza | Valor |
|---|---|
| FPS informado pelo container | 60,0772 |
| Mediana dos intervalos | 16,6667 ms (60,000 fps) |
| Intervalos a mais de 20% da mediana | 1 — frame 0→1, **2,683 ms** |
| Efeito no trecho (carimbo − nominal) | **+9,3 ms** em 7,23 s (0,13%) |

**Veredito: CFR confiável**, com o mesmo artefato do J_Correndo no primeiro intervalo
(aqui curto, lá longo). O efeito no trecho é maior que no J_Correndo (+9,3 ms contra
−2,0 ms), mas continua bem abaixo da incerteza do cronômetro manual.

**Atualização 2026-10-17:** a base oficial passou a ser o carimbo
(`analisar_cruzamentos.carregar_keypoints`). A taxa de 60,077 do contêiner vem da média
distorcida pelo 1º intervalo; os carimbos dão 60,000 fps. Com a base nova: 7,2374 s e
2,7634 m/s (−0,10% contra o cronômetro), antes 7,2281 s e 2,7670 m/s (+0,03%).

---

## 2. Versão comprimida (848×478) — registro

O primeiro arquivo recebido tinha 848×478 (versão comprimida do mesmo vídeo). Com ele:
corredor com ~60 px de altura, pessoa detectada em 32% dos frames, 30,8% de
disponibilidade no trecho, buraco de até 1,37 s e gap de saída de 400 ms → **inválido**.
Mesma causa do J_Correndo com `imgsz` 640: poucos pixels no corredor. Conclusão para o
protocolo: **analisar sempre o arquivo original da câmera**, nunca uma cópia enviada por
aplicativo de mensagem. Os metadados dessa extração estão em
`resultados/keypoints/J_Variando_imgsz864.json`.

---

## 3. Resultado com o original (1920×1080, imgsz 1920)

- Pessoa detectada em 653/653 frames; 100% de disponibilidade no trecho; entrada e
  saída interpoladas entre frames consecutivos.
- σx do centro dos ombros: 0,42 px (≈ 0,6 cm).
- Tempo 7,2281 s contra 7,23 s do cronômetro (−1,9 ms); V absoluta 2,7670 m/s
  (+0,03%); média das V instantâneas com sinal 2,7674 m/s (+0,02% da V absoluta).
- V instantânea bruta entre 0,66 e 6,70 m/s (fora o frame 279): a variação de
  velocidade pedida ao corredor aparece claramente no gráfico.

## 4. Falso positivo no frame 279

No frame 279 o YOLO detecta uma segunda "pessoa" na grade da escada à direita
(bbox 99×113 px, área maior que a do corredor, ~45×139 px). A regra "maior bbox"
escolhe essa detecção por um único frame: o centro dos ombros salta de x≈921 para
x≈1559 e volta. A um frame (17 ms; 33 ms a 0,5x) isso passa despercebido no vídeo.

Efeito, sem descartar nada:
- V absoluta: nenhum (os cruzamentos são nos frames 37–38 e 472–473).
- Média das V instantâneas com sinal: o salto de ida (−509 m/s) e o de volta
  (+513 m/s) se cancelam; sem eles a média seria 2,7721 m/s em vez de 2,7674 m/s.
- Média com |Δx| (legado): **5,12 m/s**, quase o dobro. O `abs()` soma os dois saltos.

É o primeiro caso observado do viés do `abs()`, e a causa é **erro de seleção da
pessoa**, não ruído de keypoint. Mostra também que a regra "maior bbox" é frágil
quando há outros objetos ou pessoas na cena. Correção possível (não aplicada):
escolher a detecção mais próxima da posição anterior do corredor — mudança de método,
a fazer como experimento comparativo.
