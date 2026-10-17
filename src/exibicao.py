"""
Utilitários de EXIBIÇÃO em tela.

Nada aqui toca em dado medido: as funções só reduzem CÓPIAS dos frames para que a
janela caiba no monitor. Quem precisar converter um clique da janela para coordenada
do vídeo original deve dividir pela escala retornada por calcular_escala_exibicao().
"""

import ctypes

import cv2

# Fração da área útil da tela que a janela pode ocupar (sobra espaço para a barra
# de título da janela e para a barra de tarefas).
MARGEM_TELA = 0.92

# Usado quando não é possível consultar o tamanho da tela (fora do Windows).
TELA_PADRAO = (1280, 720)


def tamanho_area_util():
    """
    Largura e altura da área útil do monitor principal, em pixels lógicos.

    No Windows com escala de exibição (ex.: 125%), uma janela OpenCV de 1920 px
    ocupa 2400 px físicos e não cabe na tela. Medir em pixels lógicos resolve isso.
    """
    try:
        usuario32 = ctypes.windll.user32
        # SM_CXFULLSCREEN / SM_CYFULLSCREEN: área de cliente de uma janela
        # maximizada, já descontando a barra de tarefas.
        return usuario32.GetSystemMetrics(16), usuario32.GetSystemMetrics(17)
    except Exception:
        return TELA_PADRAO


def calcular_escala_exibicao(largura_frame, altura_frame):
    """Fator (≤ 1) que faz um frame largura×altura caber na tela."""
    largura_tela, altura_tela = tamanho_area_util()

    return min(
        1.0,
        MARGEM_TELA * largura_tela / largura_frame,
        MARGEM_TELA * altura_tela / altura_frame,
    )


def reduzir_para_exibicao(frame, escala):
    """Cópia reduzida do frame. Só para a tela; nunca use o retorno para medir."""
    if escala >= 1.0:
        return frame

    return cv2.resize(
        frame,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_AREA,
    )
