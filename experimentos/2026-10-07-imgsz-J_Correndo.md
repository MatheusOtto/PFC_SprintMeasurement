# Experimento — resolução de entrada do YOLO (`imgsz`) no J_Correndo

Data: 2026-10-07
Vídeo: `videos/J_Correndo.mp4` — 1920×1080, 59,979 fps (gravado como 60 fps), 807 frames.
Montagem: câmera no nível do solo a 21 m da linha de corrida; zona de **20 m**.
Referência: cronômetro manual, **3,48 s** → V_ref = 20 / 3,48 = **5,747 m/s**.

## Motivação

No diagnóstico de keypoints deste vídeo (`diagnostico/J_Correndo.md`) o gargalo foi a
**detecção da pessoa**, não os keypoints: com `imgsz` padrão (640) o corredor, de ~137 px
de altura no vídeo, chega ao modelo com ~46 px; a confiança da pessoa fica colada no limiar
padrão (0,25) e só 30% dos frames da passagem têm detecção. A linha de saída fica sem
medição válida próxima e o cruzamento seria interpolado sobre 3,15 s.

> Nota de honestidade: a hipótese abaixo foi formulada **depois** de ver a variante 640
> (ela é a observação que motivou o experimento). As variantes 1280 e 1920 não tinham sido
> rodadas quando hipótese, métrica e critério foram escritos.

## 1. Hipótese (declarada antes de medir)

Aumentar `imgsz` de 640 para 1280 e 1920 **aumenta a disponibilidade de medição válida dos
ombros no trecho de 20 m**, porque o corredor passa de ~46 px para ~91 px e ~137 px de
altura na entrada do modelo. Com isso, os intervalos usados na interpolação dos dois
cruzamentos devem cair abaixo de `GAP_MAX_CRUZAMENTO_S` (0,25 s).

**Concluo que a mudança é ruim se:**
- a disponibilidade no trecho não subir pelo menos 10 pontos percentuais; ou
- aparecerem outras pessoas selecionadas no lugar do corredor (saltos na trajetória); ou
- o ruído do centro dos ombros (σx) aumentar.

## 2. Métrica (definida antes de rodar)

Primárias, porque respondem diretamente "dá para medir os 20 m?":
1. **Disponibilidade no trecho** = frames com os dois ombros ≥ 0,35 / frames entre o
   cruzamento de entrada e o de saída (mesma definição do relatório legado).
2. **Gap de entrada e de saída** (ms) e se cada um respeita `GAP_MAX_CRUZAMENTO_S`.
3. **Maior gap dentro do trecho** (ms).

Secundárias:
- σx do centro dos ombros no trecho (px), por segunda diferença em frames consecutivos.
- Tempo de processamento por frame (custo da variante).
- V absoluta vs cronômetro. **Só informativa**: n = 1 ensaio e o cronômetro manual tem
  incerteza da ordem de 0,1–0,2 s (3–6% em 3,48 s). Não serve para escolher vencedor.

## 3. Procedimento

Única variável: `imgsz` ∈ {640, 1280, 1920}. Mantidos fixos: modelo `yolo26n-pose.pt`
(md5 `af743a72087d8101c1025b9165056f79`), `conf=0.25` explícito, seleção da maior bbox,
limiar de ombro 0,35, ambiente `.venv` (versões no JSON de cada extração), mesma
calibração para as três variantes.

Calibração **provisória**: centro das duas garrafas brancas no frame 0, localizadas por
inspeção ampliada + centroide de pixels claros — x = 263 e x = 1766 px (±2 px), 1503 px
para 20 m. Ela afeta o valor absoluto de V, mas é a mesma nas três variantes, então não
afeta a comparação. Deve ser confirmada com `src/calibrar.py`.

```
python src/extrair_keypoints.py --imgsz 640
python src/extrair_keypoints.py --imgsz 1280
python src/extrair_keypoints.py --imgsz 1920
python src/analisar_cruzamentos.py resultados/keypoints/J_Correndo_imgsz640.csv \
    resultados/keypoints/J_Correndo_imgsz1280.csv resultados/keypoints/J_Correndo_imgsz1920.csv \
    --calibracao calibracao/J_Correndo.json --t-cronometro 3.48
```

## 4. Resultados

Um único vídeo, logo uma linha por variante. Dados brutos em `resultados/keypoints/`
(CSV + JSON com versões); saída completa do analisador reproduzível com o comando acima.

| | imgsz 640 (legado) | imgsz 1280 | imgsz 1920 (nativo) |
|---|---|---|---|
| Altura do corredor na entrada do modelo | ~46 px | ~91 px | ~137 px |
| Frames com pessoa (vídeo todo) | 134 / 807 | 558 / 807 | 693 / 807 |
| Medições válidas dos ombros (vídeo todo) | 122 | 449 | 459 |
| **Gap de entrada** | 233 ms (frames 282→296) | **17 ms** (291→292) | **17 ms** (291→292) |
| **Gap de saída** | **3151 ms** (479→668) | **17 ms** (495→496) | **17 ms** (495→496) |
| **Disponibilidade no trecho** | 13,4% (50/373) | **100%** (205/205) | **100%** (205/205) |
| Maior gap dentro do trecho | 3151 ms | 17 ms | 17 ms |
| σx centro dos ombros | 1,46 px (n = 23) | 0,82 px (n = 204) | **0,57 px** (n = 204) |
| Frames com > 1 pessoa no trecho | 0 | 4 | 3 |
| Processamento (CPU) | 65 ms/frame | 145 ms/frame | 292 ms/frame |
| Tempo no trecho (FPS nominal) | 4,1501 s | 3,4007 s | 3,4053 s |
| V absoluta | 4,819 m/s | 5,881 m/s | 5,873 m/s |
| Erro V vs cronômetro (5,747 m/s) | −16,2% | +2,3% | +2,2% |
| Status (gap ≤ 250 ms) | **INVÁLIDO** | válido | válido |

Verificações complementares:

- **Seleção da pessoa.** No trecho, todo deslocamento entre frames consecutivos tem o
  sentido esperado (entre −3,4 e −11,9 px/frame) e nenhum foge mais de 6σ da mediana:
  a maior bbox foi sempre o corredor, também nos 3–4 frames com mais de uma pessoa.
  Fora do trecho, com 1920 o modelo passa a ver mais de uma pessoa em 135 frames antes
  da entrada (quase todos entre 0 e 232) e em 6 depois da saída — provavelmente pessoas ao
  fundo.
- **Concordância entre 1280 e 1920.** Centro dos ombros, frame a frame no trecho:
  diferença média −0,84 px, desvio 1,24 px, máximo 4,7 px (n = 206). Instantes de
  cruzamento diferem 5,7 ms (entrada) e 1,1 ms (saída); tempo do trecho, 4,6 ms (0,13%).
- **Base de tempo.** Carimbo real − FPS nominal = −2,0 ms no trecho (0,06%) nas variantes
  válidas. Desprezível neste vídeo (ver `diagnostico/J_Correndo.md`, seção 1).
- **Sensibilidade ao `GAP_MAX_CRUZAMENTO_S`.** Com 1280 e 1920 os dois gaps são de 17 ms
  e passam em qualquer limite ≥ 17 ms. Com 640 a saída (3151 ms) falha em qualquer limite
  abaixo de 3,1 s. A conclusão não depende do valor escolhido para o limite.
- **Reprodução do legado.** A variante 640 do extrator novo é idêntica ao backlog de
  `Testar_Keypoints_YOLO.py` do mesmo dia (mesmas detecções; diferença máxima 6·10⁻⁵ px).

## 5. Conclusão

**Hipótese confirmada.** Com o corredor a 21 m, `imgsz` 640 torna o trecho de 20 m
**não mensurável**: o cruzamento de saída fica sem medição válida e a interpolação
atravessa 3,15 s, misturando a corrida com a pessoa caminhando de volta (−16% contra o
cronômetro). Com 1280 ou 1920 a disponibilidade no trecho vai a 100%, os dois
cruzamentos são interpolados entre frames consecutivos e o ruído dos ombros cai.
Nenhum critério de falha foi atingido: disponibilidade +87 p.p., nenhum troca de pessoa
no trecho, σx menor.

**Entre 1280 e 1920 não há vencedor nas métricas primárias**: as duas têm 100% de
disponibilidade e gaps de 17 ms, e a diferença de V (0,13%) está dentro do ruído. A
diferença está numa métrica secundária: σx 0,57 px contra 0,82 px (n = 204 em cada).

**Decisão:** o extrator novo passa a usar a **resolução nativa** do vídeo por padrão
(`IMGSZ = "nativo"`, 1920 neste vídeo). Motivos: (1) empata na métrica primária;
(2) menor ruído de keypoint; (3) é a escolha coerente com processamento offline
(problema conhecido 5 do README); (4) "nativo" e não 1920 fixo, para não ampliar vídeos 720p, o que
seria outra variável. Custo: ~4 min de CPU para este vídeo, irrelevante offline.

**Sobre o erro de +2,2% contra o cronômetro: inconclusivo.** É um único ensaio, e a
incerteza do cronômetro manual (da ordem de 0,1–0,2 s, 3–6% em 3,48 s) é maior que a
discrepância observada (0,075 s). Não se pode afirmar nem que o software está enviesado
nem que acerta.

## 6. Limitações do próprio experimento

- Um único vídeo, um único ensaio, uma única condição de luz e de distância.
- Calibração provisória (por inspeção da imagem, não por cliques). Afeta a V absoluta,
  não a comparação entre variantes.
- σx por segunda diferença inclui oscilação real do tronco e aceleração: é um limite
  superior do ruído do keypoint, comparável entre variantes mas não um valor absoluto.
- Cada variante rodou uma vez. Determinismo em CPU assumido; sustentado pela variante
  640 reproduzir o backlog do mesmo dia.
- Com 1920 o modelo detecta mais pessoas ao fundo. Aqui a maior bbox resolveu, mas em
  cenas com alguém mais perto da câmera que o corredor a regra falharia. Um critério de
  seleção mais robusto (continuidade de trajetória) fica para quando aparecer o caso.
- Só o modelo nano foi testado. Modelos maiores (problema conhecido 5 do README) são outro
  experimento.

## 7. Revisão crítica

A revisão abaixo segue um checklist de metrologia: erro sistemático, seleção de dados, critérios e reprodutibilidade.

1. **Erro sistemático disfarçado de ruído.** O +2,2% tem candidatos mecânicos que não
   somem com mais ensaios:
   - **Profundidade da raia diferente da linha das garrafas** (problema conhecido 3 do README).
     A escala vale para a profundidade das garrafas. A 21 m da câmera, se o corredor
     passou 1 m mais perto, a V sai ~4,8% maior; os +2,2% equivalem a ~0,46 m.
     Informação do autor: o corredor largou ao lado da garrafa, então no início a raia
     coincide com a linha das garrafas; esta hipótese só se sustenta se ele tiver
     derivado em profundidade durante a corrida (não verificado). Fica enfraquecida,
     não descartada. **Se for investigada, implica protocolo de coleta**: marcar a raia.
   - Câmera no nível do solo olhando para cima: a vertical que passa pela garrafa pode
     não ser vertical na imagem na altura dos ombros (inclinação da câmera).
   - Calibração ±2 px por ponto → ±0,19% na escala. Não explica 2,2% sozinha.
   - Base de tempo: 0,06%. Interpolação entre frames consecutivos: desprezível.
2. **Suposições não declaradas.** O corredor corre na profundidade das marcações; a
   câmera não tem inclinação nem rotação; a maior bbox é o corredor.
3. **Propagação de incerteza.** Não há estimativa formal. Componentes já quantificados:
   escala ±0,19%; tempo por ruído de keypoint ~1–2 ms por cruzamento (σx/v ≈ 0,6 px ÷
   444 px/s); base de tempo 2 ms. Falta: profundidade (dominante, não medida) e cronômetro.
4. **Seleção de dados.** Com 640, a detecção some nos frames de maior velocidade e
   possivelmente numa fase da passada: descarte correlacionado com a grandeza medida.
   Com 1280/1920 não há descarte no trecho deste ensaio.
5. **Comparação entre estimadores.** Só V absoluta foi calculada; a V instantânea fica
   pendente (problemas conhecidos 2 e 4 do README). Observação para lá: entre frames consecutivos, o
   deslocamento do centro dos ombros varia de 3,4 a 11,9 px/frame (≈ 2,7 a 9,5 m/s).
   Isso é oscilação da passada mais ruído, e mostra que a V instantânea frame a frame
   é bruta e oscilante. Decisão do autor (2026-10-07): ela fica sem filtro, e a oscilação
   é discutida no texto como movimento real do ponto medido.

## 8. Atualização — calibração confirmada por cliques (mesmo dia)

`src/calibrar.py` com cliques do autor: x = 263,5 e 1767,5 px (1504 px; resolução de
clique 1,48 px). A estimativa provisória era 263 e 1766 (1503 px): diferença de 0,07% na
escala. A V instantânea foi implementada em `analisar_cruzamentos.py` nesta mesma data.

| | imgsz 640 | imgsz 1280 | imgsz 1920 |
|---|---|---|---|
| Tempo no trecho | 4,1512 s | 3,4033 s | **3,4110 s** |
| V absoluta | 4,8179 m/s | 5,8766 m/s | **5,8634 m/s** |
| Média das V instantâneas (com sinal) | 5,7995 m/s (n = 48) | 5,8759 m/s (n = 203) | **5,8698 m/s** (n = 204) |
| Média das V instantâneas (\|Δx\|, legado) | idêntica | idêntica | idêntica |
| Pares com deslocamento negativo | 0 | 0 | 0 |
| Média V inst − V absoluta | +20,4% | −0,01% | +0,11% |
| Erro V absoluta vs cronômetro | −16,2% (inválido) | +2,25% | **+2,02%** |

Conclusões do experimento inalteradas. Duas observações novas:

- **O viés do `abs(Δx)` é zero neste vídeo.** Nenhum par teve deslocamento negativo: o
  menor deslocamento entre frames consecutivos (~3,4 px) é bem maior que o ruído do
  keypoint (σx ≈ 0,6 px), então o ruído nunca inverte o sinal e o `abs()` não retifica
  nada. O viés previsto no problema conhecido 2 do README é real, mas só aparece quando o ruído é
  comparável ao deslocamento por frame (corredor lento, baixa resolução, fps alto).
  Os dois estimadores continuam calculados lado a lado.
- **Com 640 a média das instantâneas (5,80 m/s) "acerta" enquanto a V absoluta erra
  −16%.** Não é mérito do estimador: as instantâneas usam só pares válidos e não veem o
  buraco de 3,15 s na saída. Concordância entre os dois estimadores (como em 1280 e 1920,
  0,1%) é um indicador de que o trecho foi bem coberto; discordância grande é alarme.

Ensaio registrado em `resultados/ensaios.csv` com a variante 1920.

## 9. Atualização — base de tempo pelo carimbo e detecção na passagem (2026-10-17)

A base de tempo oficial passou a ser o carimbo de cada quadro (ver
`experimentos/2026-10-07-kalman.md`, seção 8). As linhas de tempo e velocidade da seção 8
ficam assim:

| | imgsz 640 | imgsz 1280 | imgsz 1920 |
|---|---|---|---|
| Tempo no trecho | 4,1488 s | 3,4013 s | **3,4090 s** |
| Velocidade média no trecho | 4,8207 m/s | 5,8801 m/s | **5,8669 m/s** |
| Diferença para o cronômetro | −16,12% (inválido) | +2,31% | **+2,08%** |
| Intervalo na saída | 3149 ms | 17 ms | 17 ms |

Entre 1280 e 1920 o tempo do trecho difere 7,7 ms (0,23%). O 4,6 ms (0,13%) da seção 4 era
da calibração provisória.
Pelo mesmo motivo, o σx da variante 1920 é 0,58 px (n = 205) com a calibração por cliques, e não
0,57 px (n = 204).

Duas métricas diferentes, que o texto antigo misturava:
- **Detecção da pessoa durante a passagem** (quadros 259–479, `pessoa_detectada`):
  29,9% (66/221) com 640; 100% (221/221) com 1280 e com 1920.
- **Disponibilidade de medição no trecho** (os dois ombros com confiança ≥ 0,35, entre os
  cruzamentos): 13,4%, 100% e 100% (seção 4).
