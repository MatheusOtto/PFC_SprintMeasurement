import cv2
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# EXPERIMENTO: VISUALIZAÇÃO E LOG DOS 17 KEYPOINTS DO YOLO POSE
# =============================================================================
#
# Objetivo:
#   Ver quais pontos corporais permanecem mais disponíveis e confiáveis
#   durante o vídeo lateral do corredor.
#
# Este script NÃO calcula velocidade e NÃO faz calibração.
# A ideia é isolar uma única variável: o comportamento dos keypoints.
#
# Comandos:
#   R = reiniciar o vídeo e limpar o backlog atual
#   Q = salvar os CSVs e sair
#
# Saídas:
#   backlog_keypoints_Andando_2.csv
#   resumo_keypoints_Andando_2.csv
# =============================================================================


# -----------------------------------------------------------------------------
# CONFIGURAÇÕES
# -----------------------------------------------------------------------------

NOME_VIDEO = "J_Correndo.mp4"

# Mantemos o mesmo modelo usado no teste anterior para não mudar outra variável.
MODELO_POSE = "yolo26n-pose.pt"

# Confiança usada apenas para classificar visualmente os pontos.
# Todos os pontos com coordenadas válidas são registrados no CSV.
LIMIAR_ALTA_CONFIANCA = 0.60
LIMIAR_MEDIA_CONFIANCA = 0.30

# Limite usado no resumo para dizer se um keypoint ficou "disponível".
LIMIAR_DISPONIBILIDADE = 0.35

# YOLO Pose / COCO: 17 keypoints.
# "Esquerdo" e "direito" são relativos ao corpo da pessoa,
# não ao lado esquerdo/direito da imagem.
NOMES_KEYPOINTS = [
    "nariz",              # 0
    "olho_esquerdo",      # 1
    "olho_direito",       # 2
    "orelha_esquerda",    # 3
    "orelha_direita",     # 4
    "ombro_esquerdo",     # 5
    "ombro_direito",      # 6
    "cotovelo_esquerdo",  # 7
    "cotovelo_direito",   # 8
    "pulso_esquerdo",     # 9
    "pulso_direito",      # 10
    "quadril_esquerdo",   # 11
    "quadril_direito",    # 12
    "joelho_esquerdo",    # 13
    "joelho_direito",     # 14
    "tornozelo_esquerdo", # 15
    "tornozelo_direito",  # 16
]

# Ligações do esqueleto apenas para facilitar a leitura visual.
# Elas NÃO entram em nenhum cálculo.
LIGACOES_ESQUELETO = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


# -----------------------------------------------------------------------------
# CAMINHO DO VÍDEO
# -----------------------------------------------------------------------------

def localizar_video():
    """
    Procura Andando_2.mp4 em locais comuns do projeto.

    Isso evita o problema anterior de depender somente da pasta atual
    de execução do Python.
    """
    if "__file__" in globals():
        pasta_script = Path(__file__).resolve().parent
    else:
        pasta_script = Path.cwd()

    pasta_execucao = Path.cwd()

    candidatos = [
        pasta_script / "videos" / NOME_VIDEO,
        pasta_script / NOME_VIDEO,
        pasta_execucao / "videos" / NOME_VIDEO,
        pasta_execucao / NOME_VIDEO,
    ]

    candidatos_unicos = []
    for caminho in candidatos:
        caminho = caminho.resolve()
        if caminho not in candidatos_unicos:
            candidatos_unicos.append(caminho)

    for caminho in candidatos_unicos:
        if caminho.exists():
            print(f"Vídeo encontrado em: {caminho}")
            return caminho

    print(f"\nERRO: {NOME_VIDEO} não foi encontrado.")
    print("Caminhos verificados:")
    for caminho in candidatos_unicos:
        print(f"  - {caminho}")

    return None


# -----------------------------------------------------------------------------
# SELEÇÃO DA PESSOA
# -----------------------------------------------------------------------------

def selecionar_pessoa_principal(resultado):
    """
    Se o YOLO encontrar mais de uma pessoa, escolhe provisoriamente
    a pessoa com a maior bounding box.

    Isso é adequado SOMENTE para este teste com um corredor principal.
    Não é ainda um algoritmo de tracking.
    """
    if resultado.boxes is None or resultado.keypoints is None:
        return None

    if len(resultado.boxes) == 0:
        return None

    caixas = resultado.boxes.xyxy.cpu().numpy()
    keypoints = resultado.keypoints.data.cpu().numpy()

    larguras = caixas[:, 2] - caixas[:, 0]
    alturas = caixas[:, 3] - caixas[:, 1]
    areas = larguras * alturas

    indice = int(np.argmax(areas))

    caixa = caixas[indice]
    pontos = keypoints[indice]

    if resultado.boxes.conf is not None:
        confianca_pessoa = float(resultado.boxes.conf[indice].cpu().item())
    else:
        confianca_pessoa = np.nan

    return caixa, pontos, confianca_pessoa


# -----------------------------------------------------------------------------
# DESENHO
# -----------------------------------------------------------------------------

def cor_por_confianca(confianca):
    """
    Classificação visual:
      verde   = confiança alta
      amarelo = confiança média
      vermelho = confiança baixa
    """
    if confianca >= LIMIAR_ALTA_CONFIANCA:
        return (0, 255, 0)       # BGR
    elif confianca >= LIMIAR_MEDIA_CONFIANCA:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


def desenhar_keypoints(frame, pontos):
    """
    Desenha todos os keypoints que possuem coordenadas válidas.

    Ao lado de cada ponto aparece:
       índice:confiança

    Exemplo:
       11:0.91
    significa keypoint 11 (quadril esquerdo) com confiança 0.91.
    """
    pontos_validos = {}

    for indice, ponto in enumerate(pontos):
        x = float(ponto[0])
        y = float(ponto[1])
        conf = float(ponto[2]) if ponto.shape[0] >= 3 else 1.0

        if x > 0 and y > 0:
            px = int(round(x))
            py = int(round(y))

            pontos_validos[indice] = (px, py, conf)

            cor = cor_por_confianca(conf)

            cv2.circle(frame, (px, py), 6, cor, -1)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)

            cv2.putText(
                frame,
                f"{indice}:{conf:.2f}",
                (px + 7, py - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                cor,
                1,
                cv2.LINE_AA,
            )

    # Desenha o esqueleto somente quando os dois pontos possuem
    # confiança pelo menos média. Isso evita linhas absurdas dominando a tela.
    for i, j in LIGACOES_ESQUELETO:
        if i in pontos_validos and j in pontos_validos:
            xi, yi, ci = pontos_validos[i]
            xj, yj, cj = pontos_validos[j]

            if ci >= LIMIAR_MEDIA_CONFIANCA and cj >= LIMIAR_MEDIA_CONFIANCA:
                cv2.line(
                    frame,
                    (xi, yi),
                    (xj, yj),
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

    return pontos_validos


def desenhar_legenda(frame):
    """Mostra a correspondência índice -> parte do corpo."""
    altura, largura = frame.shape[:2]

    x0 = 10
    y0 = 25
    espacamento = 18

    # Fundo escuro para facilitar a leitura sem esconder muito do vídeo.
    largura_painel = min(260, largura - 20)
    altura_painel = min(17 * espacamento + 45, altura - 20)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (5, 5),
        (5 + largura_painel, 5 + altura_painel),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(
        frame,
        "KEYPOINTS YOLO POSE",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    for indice, nome in enumerate(NOMES_KEYPOINTS):
        y = y0 + 22 + indice * espacamento

        if y > altura - 15:
            break

        cv2.putText(
            frame,
            f"{indice:02d} - {nome}",
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# -----------------------------------------------------------------------------
# BACKLOG
# -----------------------------------------------------------------------------

def criar_cabecalho_csv():
    cabecalho = [
        "frame",
        "timestamp_s",
        "pessoa_detectada",
        "confianca_pessoa",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
    ]

    for indice, nome in enumerate(NOMES_KEYPOINTS):
        cabecalho.extend([
            f"kp{indice}_{nome}_x",
            f"kp{indice}_{nome}_y",
            f"kp{indice}_{nome}_conf",
        ])

    return cabecalho


def criar_linha_backlog(
    frame_num,
    timestamp_s,
    caixa=None,
    pontos=None,
    confianca_pessoa=np.nan,
):
    """
    Gera uma linha do CSV para TODOS os frames.
    Se a pessoa não for detectada, os campos dos pontos ficam vazios.
    """
    if caixa is None or pontos is None:
        linha = [
            frame_num,
            timestamp_s,
            0,
            "",
            "", "", "", "",
        ]

        for _ in NOMES_KEYPOINTS:
            linha.extend(["", "", ""])

        return linha

    x1, y1, x2, y2 = caixa

    linha = [
        frame_num,
        timestamp_s,
        1,
        confianca_pessoa,
        x1, y1, x2, y2,
    ]

    for ponto in pontos:
        x = float(ponto[0])
        y = float(ponto[1])
        conf = float(ponto[2]) if ponto.shape[0] >= 3 else 1.0

        linha.extend([x, y, conf])

    return linha


def salvar_backlog(linhas, pasta_saida):
    caminho = pasta_saida / "backlog_keypoints.csv"

    with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(criar_cabecalho_csv())
        escritor.writerows(linhas)

    print(f"Backlog salvo em: {caminho}")
    return caminho


def gerar_resumo(linhas, pasta_saida):
    """
    Gera um resumo simples de DISPONIBILIDADE e CONFIANÇA.

    Importante:
    Não usamos aqui 'desvio de X' como medida de estabilidade, porque o atleta
    está realmente se deslocando. Essa análise da trajetória será feita depois
    sobre o backlog bruto.
    """
    total_frames = len(linhas)

    if total_frames == 0:
        return None

    resumo = []

    # O primeiro keypoint começa na coluna 8.
    inicio_keypoints = 8

    for indice, nome in enumerate(NOMES_KEYPOINTS):
        coluna_x = inicio_keypoints + indice * 3
        coluna_y = coluna_x + 1
        coluna_conf = coluna_x + 2

        confiancas = []
        coordenadas_validas = 0
        disponiveis = 0
        maior_sequencia_perdida = 0
        sequencia_perdida_atual = 0

        for linha in linhas:
            x = linha[coluna_x]
            y = linha[coluna_y]
            conf = linha[coluna_conf]

            valido = (
                x != ""
                and y != ""
                and conf != ""
                and float(x) > 0
                and float(y) > 0
            )

            if valido:
                coordenadas_validas += 1
                conf_float = float(conf)
                confiancas.append(conf_float)

                if conf_float >= LIMIAR_DISPONIBILIDADE:
                    disponiveis += 1
                    sequencia_perdida_atual = 0
                else:
                    sequencia_perdida_atual += 1
            else:
                sequencia_perdida_atual += 1

            maior_sequencia_perdida = max(
                maior_sequencia_perdida,
                sequencia_perdida_atual,
            )

        if confiancas:
            conf_media = float(np.mean(confiancas))
            conf_mediana = float(np.median(confiancas))
            conf_min = float(np.min(confiancas))
        else:
            conf_media = np.nan
            conf_mediana = np.nan
            conf_min = np.nan

        taxa_disponibilidade = 100.0 * disponiveis / total_frames

        resumo.append([
            indice,
            nome,
            total_frames,
            coordenadas_validas,
            disponiveis,
            taxa_disponibilidade,
            conf_media,
            conf_mediana,
            conf_min,
            maior_sequencia_perdida,
        ])

    caminho = pasta_saida / "resumo_keypoints.csv"

    with open(caminho, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

        escritor.writerow([
            "indice",
            "keypoint",
            "total_frames",
            "frames_com_coordenada",
            f"frames_conf_maior_igual_{LIMIAR_DISPONIBILIDADE}",
            "taxa_disponibilidade_percentual",
            "confianca_media",
            "confianca_mediana",
            "confianca_minima",
            "maior_sequencia_frames_indisponivel",
        ])

        escritor.writerows(resumo)

    print(f"Resumo salvo em: {caminho}")
    return caminho


def salvar_resultados(linhas, pasta_saida):
    if len(linhas) == 0:
        print("Nenhum frame registrado; nada para salvar.")
        return

    salvar_backlog(linhas, pasta_saida)
    gerar_resumo(linhas, pasta_saida)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

caminho_video = localizar_video()

if caminho_video is None:
    raise SystemExit

# Os CSVs serão salvos ao lado do vídeo.
pasta_saida = caminho_video.parent

modelo = YOLO(MODELO_POSE)

cap = cv2.VideoCapture(str(caminho_video))

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {caminho_video}")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("FPS inválido.")
    cap.release()
    raise SystemExit

print(f"FPS detectado: {fps:.3f}")
print("\nComandos:")
print("  R = reiniciar vídeo e limpar backlog")
print("  Q = salvar CSVs e sair\n")

linhas_backlog = []
frame_num = 0

executar = True

while executar:
    ret, frame = cap.read()

    # -------------------------------------------------------------------------
    # FIM DO VÍDEO
    # -------------------------------------------------------------------------
    if not ret:
        salvar_resultados(linhas_backlog, pasta_saida)

        print("\nFim do vídeo.")
        print("R = rodar novamente e limpar o backlog")
        print("Q = sair")

        while True:
            tecla = cv2.waitKey(30) & 0xFF

            if tecla == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                linhas_backlog = []
                frame_num = 0
                print("\nVídeo reiniciado. Backlog anterior limpo.")
                break

            elif tecla == ord("q"):
                executar = False
                break

        continue

    timestamp_s = frame_num / fps

    # -------------------------------------------------------------------------
    # INFERÊNCIA YOLO POSE
    # -------------------------------------------------------------------------
    resultados = modelo.predict(frame, verbose=False)
    resultado = resultados[0]

    pessoa = selecionar_pessoa_principal(resultado)

    if pessoa is not None:
        caixa, pontos, confianca_pessoa = pessoa

        x1, y1, x2, y2 = caixa.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        pontos_validos = desenhar_keypoints(frame, pontos)

        linhas_backlog.append(
            criar_linha_backlog(
                frame_num=frame_num,
                timestamp_s=timestamp_s,
                caixa=caixa,
                pontos=pontos,
                confianca_pessoa=confianca_pessoa,
            )
        )

        cv2.putText(
            frame,
            f"Pessoa conf: {confianca_pessoa:.2f} | pontos visiveis: {len(pontos_validos)}/17",
            (280, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    else:
        linhas_backlog.append(
            criar_linha_backlog(
                frame_num=frame_num,
                timestamp_s=timestamp_s,
            )
        )

        cv2.putText(
            frame,
            "PESSOA NAO DETECTADA",
            (280, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # -------------------------------------------------------------------------
    # INTERFACE
    # -------------------------------------------------------------------------
    desenhar_legenda(frame)

    cv2.putText(
        frame,
        f"Frame: {frame_num} | t = {timestamp_s:.2f}s",
        (280, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "R = reiniciar | Q = salvar e sair",
        (280, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("Teste de Keypoints - YOLO Pose", frame)

    tecla = cv2.waitKey(30) & 0xFF

    if tecla == ord("q"):
        salvar_resultados(linhas_backlog, pasta_saida)
        executar = False

    elif tecla == ord("r"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        linhas_backlog = []
        frame_num = 0
        print("\nVídeo reiniciado. Backlog atual apagado.")
        continue

    frame_num += 1


cap.release()
cv2.destroyAllWindows()
