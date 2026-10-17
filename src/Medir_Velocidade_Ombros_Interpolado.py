import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# SISTEMA DE MEDIÇÃO DE VELOCIDADE - CENTRO DOS OMBROS
# =============================================================================
#
# Alterações desta versão:
#
# 1) Removido o debounce de 3 frames para determinar entrada/saída.
#    Agora o sistema detecta o cruzamento geométrico das linhas de 5 m.
#
# 2) O instante de cruzamento é estimado por INTERPOLAÇÃO LINEAR entre
#    duas medições válidas consecutivas do centro dos ombros.
#
# 3) Relatório final detalhado:
#       - resolução e FPS
#       - escala pixel -> metro
#       - frame anterior/posterior ao cruzamento
#       - frame interpolado de entrada/saída
#       - tempos interpolados
#       - frames no trecho
#       - medições válidas
#       - ombros indisponíveis
#       - disponibilidade
#       - velocidades com 4 casas
#       - diferença entre V absoluta e média das instantâneas
#
# 4) R reinicia o teste sem refazer a calibração.
#
# IMPORTANTE:
# A velocidade instantânea ainda mantém abs(delta_x), propositalmente.
# O tratamento formal do ruído será uma etapa posterior.
# =============================================================================


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

NOME_VIDEO = "Trotando_2.mp4"

modelo_pose = YOLO("yolo26n-pose.pt")

# Distância REAL entre as duas marcações clicadas no vídeo.
DISTANCIA_REAL_METROS = 5.0

# COCO Pose:
# 5 = ombro esquerdo
# 6 = ombro direito
INDICE_OMBRO_ESQUERDO = 5
INDICE_OMBRO_DIREITO = 6

# Os dois ombros precisam satisfazer esse limiar.
CONFIANCA_MIN_OMBRO = 0.35

# Média móvel usada SOMENTE para exibição em tela.
TAMANHO_FILTRO_EXIBICAO = 7

LIMITE_AMOSTRAS = 10000

pontos_calibracao = []


# =============================================================================
# VÍDEO
# =============================================================================

def localizar_video():
    """
    Procura o vídeo em locais comuns para não depender da pasta de execução
    do VS Code/Spyder.
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
            return str(caminho)

    print(f"\nERRO: não encontrei {NOME_VIDEO}.")
    print("Caminhos verificados:")

    for caminho in candidatos_unicos:
        print(f"  - {caminho}")

    return None


# =============================================================================
# CALIBRAÇÃO
# =============================================================================

def capturar_cliques(event, x, y, flags, param):
    """
    Salva os dois pontos cujas posições longitudinais estão separadas
    por exatamente 5 metros.
    """
    global pontos_calibracao

    if event == cv2.EVENT_LBUTTONUP:
        if len(pontos_calibracao) < 2:
            pontos_calibracao.append((x, y))
            print(f"Marcação {len(pontos_calibracao)}: X={x}, Y={y}")


# =============================================================================
# YOLO POSE
# =============================================================================

def obter_centro_ombros_yolo(frame):
    """
    Retorna o centro entre os dois ombros do corredor.

    Só há medição válida se:
      - a pessoa for detectada;
      - os dois ombros tiverem coordenadas válidas;
      - os dois ombros tiverem confiança >= CONFIANCA_MIN_OMBRO.

    Retorno:
        cx, cy,
        conf_esq, conf_dir,
        ponto_esq, ponto_dir,
        caixa

    ou None.
    """
    resultados = modelo_pose.predict(frame, verbose=False)
    resultado = resultados[0]

    if resultado.boxes is None or resultado.keypoints is None:
        return None

    if len(resultado.boxes) == 0:
        return None

    caixas = resultado.boxes.xyxy.cpu().numpy()
    keypoints = resultado.keypoints.data.cpu().numpy()

    # Para o experimento atual há um corredor principal.
    # Se houver mais pessoas, escolhemos provisoriamente a maior bounding box.
    larguras = caixas[:, 2] - caixas[:, 0]
    alturas = caixas[:, 3] - caixas[:, 1]
    areas = larguras * alturas

    indice_pessoa = int(np.argmax(areas))

    pontos = keypoints[indice_pessoa]

    ombro_esq = pontos[INDICE_OMBRO_ESQUERDO]
    ombro_dir = pontos[INDICE_OMBRO_DIREITO]

    x_esq = float(ombro_esq[0])
    y_esq = float(ombro_esq[1])
    conf_esq = float(ombro_esq[2])

    x_dir = float(ombro_dir[0])
    y_dir = float(ombro_dir[1])
    conf_dir = float(ombro_dir[2])

    if x_esq <= 0 or y_esq <= 0 or x_dir <= 0 or y_dir <= 0:
        return None

    if conf_esq < CONFIANCA_MIN_OMBRO or conf_dir < CONFIANCA_MIN_OMBRO:
        return None

    cx = (x_esq + x_dir) / 2.0
    cy = (y_esq + y_dir) / 2.0

    caixa = caixas[indice_pessoa]

    return (
        cx,
        cy,
        conf_esq,
        conf_dir,
        (int(round(x_esq)), int(round(y_esq))),
        (int(round(x_dir)), int(round(y_dir))),
        caixa,
    )


# =============================================================================
# CRUZAMENTO E INTERPOLAÇÃO
# =============================================================================

def interpolar_cruzamento(
    x_anterior,
    frame_anterior,
    x_atual,
    frame_atual,
    x_linha,
    fps
):
    """
    Estima o instante em que o centro dos ombros cruzou uma linha vertical.

    Exemplo:
        frame 100 -> x = 493
        frame 101 -> x = 506
        linha     -> x = 500

    Fração do intervalo até a linha:

        alfa = (500 - 493) / (506 - 493)

    Então:

        frame_interpolado = 100 + alfa * (101 - 100)

    e:

        tempo = frame_interpolado / FPS

    Retorna:
        tempo_cruzamento,
        frame_interpolado,
        alfa
    """
    delta_x = x_atual - x_anterior

    if abs(delta_x) < 1e-12:
        return None

    alfa = (x_linha - x_anterior) / delta_x

    # O cruzamento precisa estar ENTRE as duas observações.
    if alfa < 0.0 or alfa > 1.0:
        return None

    frame_interpolado = (
        frame_anterior
        + alfa * (frame_atual - frame_anterior)
    )

    tempo_cruzamento = frame_interpolado / fps

    return tempo_cruzamento, frame_interpolado, alfa


# =============================================================================
# ESTADO DO ENSAIO
# =============================================================================

def criar_estado_inicial():
    return {
        # 0 = aguardando
        # 1 = dentro dos 5 m
        # 2 = finalizado
        "estado_corrida": 0,

        # Última medição válida, inclusive antes de entrar na zona.
        "cx_anterior": None,
        "frame_anterior": None,

        # Direção:
        # +1 = esquerda -> direita
        # -1 = direita -> esquerda
        "direcao": None,

        # Linha de entrada/saída
        "x_linha_entrada": None,
        "x_linha_saida": None,

        # Cruzamento de entrada
        "frame_antes_entrada": None,
        "frame_depois_entrada": None,
        "frame_entrada_interpolado": None,
        "tempo_entrada": None,
        "gap_entrada_frames": None,

        # Cruzamento de saída
        "frame_antes_saida": None,
        "frame_depois_saida": None,
        "frame_saida_interpolado": None,
        "tempo_saida": None,
        "gap_saida_frames": None,

        # Resultado temporal
        "tempo_final": 0.0,

        # Velocidade
        "velocidade_media_absoluta": 0.0,
        "media_das_instantaneas": 0.0,
        "buffer_velocidades": [],
        "historico_velocidades": [],

        # Diagnóstico de disponibilidade DURANTE o trecho.
        "medicoes_validas_trecho": 0,
        "medicoes_indisponiveis_trecho": 0,

        # Controle geral
        "frame_atual": -1,
        "frame_final": None,

        # Impede imprimir o relatório mais de uma vez
        "relatorio_impresso": False,
    }


def reiniciar_teste(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("\n======================================")
    print("TESTE REINICIADO")
    print("R = reiniciar | Q = sair")
    print("======================================\n")

    return criar_estado_inicial()


# =============================================================================
# RELATÓRIO
# =============================================================================

def imprimir_relatorio(
    teste,
    fps_video,
    largura_frame,
    altura_frame,
    distancia_pixels,
    metros_por_pixel
):
    """
    Imprime informações suficientes para diagnosticar cada execução.
    """

    if teste["relatorio_impresso"]:
        return

    teste["relatorio_impresso"] = True

    total_obs = (
        teste["medicoes_validas_trecho"]
        + teste["medicoes_indisponiveis_trecho"]
    )

    if total_obs > 0:
        disponibilidade = (
            100.0
            * teste["medicoes_validas_trecho"]
            / total_obs
        )
    else:
        disponibilidade = 0.0

    diferenca_v = (
        teste["media_das_instantaneas"]
        - teste["velocidade_media_absoluta"]
    )

    if teste["velocidade_media_absoluta"] != 0:
        diferenca_v_percentual = (
            100.0
            * diferenca_v
            / teste["velocidade_media_absoluta"]
        )
    else:
        diferenca_v_percentual = 0.0

    intervalos_instantaneos = len(teste["historico_velocidades"])

    print("\n")
    print("=" * 65)
    print("RELATÓRIO FINAL DO ENSAIO")
    print("=" * 65)

    print("\n--- VÍDEO ---")
    print(f"Resolução:                   {largura_frame} x {altura_frame} px")
    print(f"FPS informado pelo arquivo: {fps_video:.4f}")
    print(f"Tempo nominal por frame:     {1000.0 / fps_video:.4f} ms")

    print("\n--- CALIBRAÇÃO ---")
    print(f"Distância real:              {DISTANCIA_REAL_METROS:.4f} m")
    print(f"Distância na imagem:         {distancia_pixels:.4f} px")
    print(f"Escala:                      {metros_por_pixel:.8f} m/pixel")

    print("\n--- ENTRADA ---")
    print(f"Direção:                     {'esquerda -> direita' if teste['direcao'] == 1 else 'direita -> esquerda'}")
    print(f"Linha de entrada:            X = {teste['x_linha_entrada']:.4f} px")
    print(f"Frame anterior:              {teste['frame_antes_entrada']}")
    print(f"Frame posterior:             {teste['frame_depois_entrada']}")
    print(f"Gap usado na interpolação:   {teste['gap_entrada_frames']} frame(s)")
    print(f"Frame interpolado:           {teste['frame_entrada_interpolado']:.4f}")
    print(f"Tempo interpolado:           {teste['tempo_entrada']:.6f} s")

    print("\n--- SAÍDA ---")
    print(f"Linha de saída:              X = {teste['x_linha_saida']:.4f} px")
    print(f"Frame anterior:              {teste['frame_antes_saida']}")
    print(f"Frame posterior:             {teste['frame_depois_saida']}")
    print(f"Gap usado na interpolação:   {teste['gap_saida_frames']} frame(s)")
    print(f"Frame interpolado:           {teste['frame_saida_interpolado']:.4f}")
    print(f"Tempo interpolado:           {teste['tempo_saida']:.6f} s")

    print("\n--- DISPONIBILIDADE NO TRECHO ---")
    print(f"Medições válidas:            {teste['medicoes_validas_trecho']}")
    print(f"Ombros indisponíveis:        {teste['medicoes_indisponiveis_trecho']}")
    print(f"Total de frames observados:  {total_obs}")
    print(f"Disponibilidade:             {disponibilidade:.2f} %")
    print(f"Intervalos de V instantânea: {intervalos_instantaneos}")

    print("\n--- RESULTADOS ---")
    print(f"Tempo final:                 {teste['tempo_final']:.6f} s")
    print(f"V. absoluta:                 {teste['velocidade_media_absoluta']:.4f} m/s")
    print(f"Média V. instantâneas:       {teste['media_das_instantaneas']:.4f} m/s")
    print(f"Diferença Vinst - Vabs:      {diferenca_v:+.4f} m/s")
    print(f"Diferença percentual:        {diferenca_v_percentual:+.3f} %")

    print("=" * 65)
    print()


# =============================================================================
# ABERTURA
# =============================================================================

caminho_do_video = localizar_video()

if caminho_do_video is None:
    raise SystemExit

cap = cv2.VideoCapture(caminho_do_video)

if not cap.isOpened():
    print(f"Erro ao abrir vídeo: {caminho_do_video}")
    raise SystemExit

fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    cap.release()
    raise ValueError("Não foi possível obter um FPS válido do vídeo.")

largura_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolução detectada: {largura_frame} x {altura_frame}")
print(f"FPS detectado: {fps_video:.4f}")


# =============================================================================
# ETAPA 1 - CALIBRAÇÃO DOS 5 METROS
# =============================================================================

ret, frame_inicial = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Não foi possível ler o primeiro frame.")

cv2.namedWindow("Calibracao")
cv2.setMouseCallback("Calibracao", capturar_cliques)

print("\nCALIBRAÇÃO")
print("1) Clique no início da zona de 5 m.")
print("2) Clique no final da zona de 5 m.")
print("3) Pressione C para confirmar.")
print("Q = sair.")

while True:
    frame_mostra = frame_inicial.copy()

    for p in pontos_calibracao:
        cv2.circle(frame_mostra, p, 6, (0, 255, 0), -1)

    if len(pontos_calibracao) == 2:
        x_min_temp = min(pontos_calibracao[0][0], pontos_calibracao[1][0])
        x_max_temp = max(pontos_calibracao[0][0], pontos_calibracao[1][0])

        cv2.line(
            frame_mostra,
            (x_min_temp, 0),
            (x_min_temp, frame_mostra.shape[0]),
            (0, 0, 255),
            2,
        )

        cv2.line(
            frame_mostra,
            (x_max_temp, 0),
            (x_max_temp, frame_mostra.shape[0]),
            (0, 0, 255),
            2,
        )

    cv2.imshow("Calibracao", frame_mostra)

    tecla = cv2.waitKey(30) & 0xFF

    if tecla == ord("c") and len(pontos_calibracao) == 2:
        break

    if tecla == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

cv2.destroyWindow("Calibracao")


p1 = pontos_calibracao[0]
p2 = pontos_calibracao[1]

distancia_pixels = abs(p2[0] - p1[0])

if distancia_pixels == 0:
    cap.release()
    cv2.destroyAllWindows()
    raise ValueError("Os dois pontos possuem o mesmo X.")

metros_por_pixel = (
    DISTANCIA_REAL_METROS
    / distancia_pixels
)

x_min = float(min(p1[0], p2[0]))
x_max = float(max(p1[0], p2[0]))

print("\nCALIBRAÇÃO CONCLUÍDA")
print(f"5 m = {distancia_pixels:.4f} pixels")
print(f"Escala = {metros_por_pixel:.8f} m/pixel")
print(f"Linhas: X={x_min:.2f} e X={x_max:.2f}")


# =============================================================================
# ETAPA 2 - MEDIÇÃO
# =============================================================================

teste = reiniciar_teste(cap)

executar = True

while executar:

    ret, frame = cap.read()

    # =========================================================================
    # FIM DO VÍDEO
    # =========================================================================
    if not ret:

        while True:

            if teste["frame_final"] is not None:
                frame_fim = teste["frame_final"].copy()

                cv2.putText(
                    frame_fim,
                    "R = REINICIAR | Q = SAIR",
                    (50, frame_fim.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "PFC - Centro dos Ombros",
                    frame_fim
                )

            tecla = cv2.waitKey(30) & 0xFF

            if tecla == ord("q"):
                executar = False
                break

            if tecla == ord("r"):
                teste = reiniciar_teste(cap)
                break

        continue


    teste["frame_atual"] += 1
    frame_atual = teste["frame_atual"]

    deteccao = obter_centro_ombros_yolo(frame)

    # =========================================================================
    # HÁ MEDIÇÃO VÁLIDA DOS OMBROS
    # =========================================================================
    if deteccao is not None:

        (
            cx,
            cy,
            conf_esq,
            conf_dir,
            ponto_esq,
            ponto_dir,
            caixa,
        ) = deteccao

        x1, y1, x2, y2 = caixa.astype(int)

        # Diagnóstico visual
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            frame,
            ponto_esq,
            7,
            (0, 255, 255),
            -1,
        )

        cv2.circle(
            frame,
            ponto_dir,
            7,
            (0, 255, 255),
            -1,
        )

        cv2.line(
            frame,
            ponto_esq,
            ponto_dir,
            (0, 255, 255),
            2,
        )

        cv2.circle(
            frame,
            (int(round(cx)), int(round(cy))),
            10,
            (255, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Ombros L:{conf_esq:.2f} R:{conf_dir:.2f}",
            (max(10, x1), max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )


        # =====================================================================
        # ESTADO 0 - AGUARDANDO PRIMEIRO CRUZAMENTO
        # =====================================================================
        if teste["estado_corrida"] == 0:

            if (
                teste["cx_anterior"] is not None
                and teste["frame_anterior"] is not None
            ):

                x_ant = teste["cx_anterior"]
                frame_ant = teste["frame_anterior"]

                # -------------------------------------------------------------
                # Entrada pela esquerda:
                # anterior estava fora à esquerda e atual chegou/entrou na zona.
                # -------------------------------------------------------------
                entrou_pela_esquerda = (
                    x_ant < x_min
                    and cx >= x_min
                    and cx <= x_max
                )

                # -------------------------------------------------------------
                # Entrada pela direita:
                # anterior estava fora à direita e atual chegou/entrou na zona.
                # -------------------------------------------------------------
                entrou_pela_direita = (
                    x_ant > x_max
                    and cx <= x_max
                    and cx >= x_min
                )

                if entrou_pela_esquerda or entrou_pela_direita:

                    if entrou_pela_esquerda:
                        teste["direcao"] = 1
                        teste["x_linha_entrada"] = x_min
                        teste["x_linha_saida"] = x_max
                    else:
                        teste["direcao"] = -1
                        teste["x_linha_entrada"] = x_max
                        teste["x_linha_saida"] = x_min

                    cruzamento = interpolar_cruzamento(
                        x_anterior=x_ant,
                        frame_anterior=frame_ant,
                        x_atual=cx,
                        frame_atual=frame_atual,
                        x_linha=teste["x_linha_entrada"],
                        fps=fps_video,
                    )

                    if cruzamento is not None:

                        (
                            tempo_entrada,
                            frame_entrada_interpolado,
                            alfa,
                        ) = cruzamento

                        teste["estado_corrida"] = 1

                        teste["tempo_entrada"] = tempo_entrada
                        teste["frame_entrada_interpolado"] = (
                            frame_entrada_interpolado
                        )

                        teste["frame_antes_entrada"] = frame_ant
                        teste["frame_depois_entrada"] = frame_atual

                        teste["gap_entrada_frames"] = (
                            frame_atual - frame_ant
                        )

                        # O frame que confirmou a entrada possui uma
                        # medição válida dentro do trecho.
                        teste["medicoes_validas_trecho"] = 1

                        # A primeira referência para velocidade dentro
                        # do trecho passa a ser a medição atual.
                        teste["cx_anterior"] = cx
                        teste["frame_anterior"] = frame_atual

                        print("\n>>> ENTRADA DETECTADA")
                        print(
                            f"    direção: "
                            f"{'esquerda -> direita' if teste['direcao'] == 1 else 'direita -> esquerda'}"
                        )
                        print(
                            f"    frame interpolado = "
                            f"{teste['frame_entrada_interpolado']:.4f}"
                        )
                        print(
                            f"    tempo = "
                            f"{teste['tempo_entrada']:.6f} s"
                        )

                        # Não sobrescrevemos novamente no fim deste bloco.
                        # Pulamos para a interface.
                        # -----------------------------------------------------
                        # Como já atualizamos a referência, continuamos.
                        # -----------------------------------------------------
                        pass

            # Se ainda não entrou, atualiza a última medição válida.
            if teste["estado_corrida"] == 0:
                teste["cx_anterior"] = cx
                teste["frame_anterior"] = frame_atual


        # =====================================================================
        # ESTADO 1 - DENTRO DOS 5 METROS
        # =====================================================================
        elif teste["estado_corrida"] == 1:

            teste["medicoes_validas_trecho"] += 1

            x_ant = teste["cx_anterior"]
            frame_ant = teste["frame_anterior"]

            # -------------------------------------------------------------
            # Primeiro verificamos se o ponto ATUAL já ultrapassou
            # a linha final na direção esperada.
            # -------------------------------------------------------------
            if teste["direcao"] == 1:
                saiu = (
                    x_ant < teste["x_linha_saida"]
                    and cx >= teste["x_linha_saida"]
                )
            else:
                saiu = (
                    x_ant > teste["x_linha_saida"]
                    and cx <= teste["x_linha_saida"]
                )

            if saiu:

                cruzamento = interpolar_cruzamento(
                    x_anterior=x_ant,
                    frame_anterior=frame_ant,
                    x_atual=cx,
                    frame_atual=frame_atual,
                    x_linha=teste["x_linha_saida"],
                    fps=fps_video,
                )

                if cruzamento is not None:

                    (
                        tempo_saida,
                        frame_saida_interpolado,
                        alfa,
                    ) = cruzamento

                    teste["tempo_saida"] = tempo_saida
                    teste["frame_saida_interpolado"] = (
                        frame_saida_interpolado
                    )

                    teste["frame_antes_saida"] = frame_ant
                    teste["frame_depois_saida"] = frame_atual

                    teste["gap_saida_frames"] = (
                        frame_atual - frame_ant
                    )

                    teste["tempo_final"] = (
                        teste["tempo_saida"]
                        - teste["tempo_entrada"]
                    )

                    if teste["tempo_final"] > 0:
                        teste["velocidade_media_absoluta"] = (
                            DISTANCIA_REAL_METROS
                            / teste["tempo_final"]
                        )

                    if len(teste["historico_velocidades"]) > 0:
                        teste["media_das_instantaneas"] = (
                            sum(teste["historico_velocidades"])
                            / len(teste["historico_velocidades"])
                        )

                    teste["estado_corrida"] = 2

                    print("\n>>> SAÍDA DETECTADA")

                    imprimir_relatorio(
                        teste=teste,
                        fps_video=fps_video,
                        largura_frame=largura_frame,
                        altura_frame=altura_frame,
                        distancia_pixels=distancia_pixels,
                        metros_por_pixel=metros_por_pixel,
                    )

            else:

                # -------------------------------------------------------------
                # VELOCIDADE INSTANTÂNEA
                #
                # Continua propositalmente usando abs(delta_x).
                #
                # Se houver frames sem os ombros, delta_frames aumenta,
                # então delta_t também aumenta.
                # -------------------------------------------------------------
                if (
                    x_ant is not None
                    and frame_ant is not None
                ):

                    delta_s_pixels = abs(cx - x_ant)
                    delta_s_metros = (
                        delta_s_pixels
                        * metros_por_pixel
                    )

                    delta_frames = (
                        frame_atual
                        - frame_ant
                    )

                    if delta_frames > 0:

                        delta_t = (
                            delta_frames
                            / fps_video
                        )

                        vel_instantanea = (
                            delta_s_metros
                            / delta_t
                        )

                        teste["buffer_velocidades"].append(
                            vel_instantanea
                        )

                        if (
                            len(teste["buffer_velocidades"])
                            > TAMANHO_FILTRO_EXIBICAO
                        ):
                            teste["buffer_velocidades"].pop(0)

                        if (
                            len(teste["historico_velocidades"])
                            < LIMITE_AMOSTRAS
                        ):
                            teste["historico_velocidades"].append(
                                vel_instantanea
                            )

                teste["cx_anterior"] = cx
                teste["frame_anterior"] = frame_atual


    # =========================================================================
    # NÃO HÁ MEDIÇÃO VÁLIDA DOS DOIS OMBROS
    # =========================================================================
    else:

        # Só contabilizamos perda para disponibilidade se a corrida já começou
        # e ainda não terminou.
        if teste["estado_corrida"] == 1:
            teste["medicoes_indisponiveis_trecho"] += 1

        cv2.putText(
            frame,
            "CENTRO DOS OMBROS INDISPONIVEL",
            (50, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )

        # IMPORTANTE:
        # Não atualizamos cx_anterior/frame_anterior.
        #
        # Exemplo:
        #   frame 100 -> medição válida
        #   frame 101 -> indisponível
        #   frame 102 -> medição válida
        #
        # A próxima velocidade usa:
        #   delta_frames = 102 - 100 = 2
        #
        # e portanto:
        #   delta_t = 2 / FPS
        #
        # Não fingimos que o frame 101 teve uma medição.


    # =========================================================================
    # INTERFACE
    # =========================================================================

    cv2.line(
        frame,
        (int(round(x_min)), 0),
        (int(round(x_min)), frame.shape[0]),
        (0, 0, 255),
        2,
    )

    cv2.line(
        frame,
        (int(round(x_max)), 0),
        (int(round(x_max)), frame.shape[0]),
        (0, 0, 255),
        2,
    )


    if teste["estado_corrida"] == 0:

        cv2.putText(
            frame,
            "AGUARDANDO CRUZAMENTO...",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )


    elif teste["estado_corrida"] == 1:

        tempo_atual = (
            frame_atual / fps_video
            - teste["tempo_entrada"]
        )

        cv2.putText(
            frame,
            f"TEMPO: {tempo_atual:.3f} s",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        if len(teste["buffer_velocidades"]) > 0:

            vel_exibicao = (
                sum(teste["buffer_velocidades"])
                / len(teste["buffer_velocidades"])
            )

            cv2.putText(
                frame,
                f"VEL ATUAL: {vel_exibicao:.3f} m/s",
                (50, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
            )


    elif teste["estado_corrida"] == 2:

        cv2.putText(
            frame,
            f"TEMPO FINAL: {teste['tempo_final']:.4f} s",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            frame,
            f"V. ABSOLUTA: {teste['velocidade_media_absoluta']:.4f} m/s",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"V. INSTANTANEAS: {teste['media_das_instantaneas']:.4f} m/s",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 100, 100),
            2,
        )


    cv2.putText(
        frame,
        "Ponto usado: centro dos ombros",
        (50, frame.shape[0] - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "R = reiniciar | Q = sair",
        (50, frame.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


    teste["frame_final"] = frame.copy()

    cv2.imshow(
        "PFC - Centro dos Ombros",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        executar = False

    elif tecla == ord("r"):
        teste = reiniciar_teste(cap)


cap.release()
cv2.destroyAllWindows()
