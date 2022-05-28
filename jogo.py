from typing import List, Dict
import numpy as np
class RegrasDeJogo():

    @staticmethod
    #identificação do caso terminal de jogo onde todos desistiram
    def todos_desistiram(historico: str, numero_jogadores: int):
        return len(historico) >= numero_jogadores and historico.endswith('N' * (numero_jogadores - 1))

    @staticmethod
    #função de pagamento que acontece ao final de uma rodada quando um vencedor é declarado.
    def get_ganho(cartas: List[str], historico: str, numero_jogadores: int) -> List[int]:
    
        jogador = len(historico) % numero_jogadores
        jogador_cartas = cartas[:numero_jogadores]
        numero_oponentes = numero_jogadores - 1
        if historico == 'N' * numero_jogadores:
            ganho_jogador = [-1] * numero_jogadores
            ganho_jogador[np.argmax(jogador_cartas)] = numero_oponentes
            return ganho_jogador

        
        elif RegrasDeJogo.todos_desistiram(historico, numero_jogadores):
            ganho_jogador = [-1] * numero_jogadores
            ganho_jogador[jogador] = numero_oponentes
        else:
            ganho_jogador = [-1] * numero_jogadores
            cartas_ativas = []
            indices_jogando = []
            for (ix, x) in enumerate(jogador_cartas):
                if 'A' in historico[ix::numero_jogadores]:
                    ganho_jogador[ix] = -2
                    cartas_ativas.append(x)
                    indices_jogando.append(ix)
            ganho_jogador[indices_jogando[np.argmax(cartas_ativas)]] = len(cartas_ativas) - 1 + numero_oponentes
        return ganho_jogador

    @staticmethod
    #identificação do restante dos casos em que o jogo termina
    def is_estado_terminal(historico: str, numero_jogadores: int) -> bool:
        
        todos_apostam = historico.endswith('A' * numero_jogadores)
        todos_agiram = (historico.find('A') > -1) and (len(historico) - historico.find('A') == numero_jogadores)
        resta_um = RegrasDeJogo.todos_desistiram(historico, numero_jogadores)
        return todos_apostam or todos_agiram or resta_um