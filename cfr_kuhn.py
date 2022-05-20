from typing import List, Dict
import random
import numpy as np
import time
tempo_execucao = time.time()
Jogadas = ['A', 'N']  # apostar/completar apostar, não apostar(desistir)/ check
BARALHO = ['9', 'T', 'J', 'Q', 'K'] # baralho, na versão de 2 jogadores somente são utilizados k,q,j


class No_Informacao():
    #função de inicialização da classe que irá compor a árvore de jogo, o nó de informação contem o arrependimento total, a soma de todos os perfis de estratégia e o número de jogadas (2), check ou bet.
    def __init__(self):
        self.regret_total = np.zeros(shape=len(Jogadas))
        self.estrategia_total = np.zeros(shape=len(Jogadas))
        self.numero_jogadas = len(Jogadas)

    #método de normalização para as jogadas serem expressas em porcentagem
    def normalizar(self, estrategia: np.array) -> np.array:
        if sum(estrategia) > 0:
            estrategia /= sum(estrategia)
        else:
            estrategia = np.array([1.0 / self.numero_jogadas] * self.numero_jogadas)
        return estrategia

    #método para a busca da estrategia a ser jogada na iteração vigente
    def get_estrategia(self, probabilidade: float) -> np.array:

        estrategia = np.maximum(0, self.regret_total)
        estrategia = self.normalizar(estrategia)

        self.estrategia_total += probabilidade * estrategia
        return estrategia

    #armazenamento do vetor estrategia total para cálculo da média
    def get_estrategia_media(self) -> np.array:
        return self.normalizar(self.estrategia_total.copy())

#classe do jogo kuhnpoker
class Jogo():
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

        
        elif Jogo.todos_desistiram(historico, numero_jogadores):
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
        resta_um = Jogo.todos_desistiram(historico, numero_jogadores)
        return todos_apostam or todos_agiram or resta_um


class Treinador():
    #inicialização do treinador do algoritmo
    def __init__(self, numero_jogadores: int):
        self.arvore_jogo: Dict[str, No_Informacao] = {}
        self.numero_jogadores = numero_jogadores
        self.cartas = [x for x in range(self.numero_jogadores + 1)]

    #algoritmo vai se situar em que parte da árvore de jogo ele está utilizando a string carta + histórico que seria algo do tipo J aa (jogador 1 possui J e aposta, jogador 2 aposta também)
    def get_informacao_bloco(self, carta_e_historico: str) -> No_Informacao:
        if carta_e_historico not in self.arvore_jogo:
            self.arvore_jogo[carta_e_historico] = No_Informacao()
        return self.arvore_jogo[carta_e_historico]


    #define a probabilidade de se alcançar aquele nó da história do jogo diante das chances de se escolher check/bet de cada jogador. produto de probabilidades de se alcançar um evento

    def get_probabilidade(self, probabilidades: np.array, jogador: int):
        return np.prod(probabilidades[:jogador]) * np.prod(probabilidades[jogador + 1:])


    #metodo principal do algoritmo counterfactural regret minimization
    def cfr(self, cartas: List[str], historico: str, probabilidade: np.array, jogador_atual: int) -> np.array:

        #verifica se o jogo acabou
        if Jogo.is_estado_terminal(historico, self.numero_jogadores):
            return Jogo.get_ganho(cartas, historico, self.numero_jogadores)
        
        minha_carta = cartas[jogador_atual]
        informacao_atual = self.get_informacao_bloco(str(minha_carta) + historico)

        estrategia = informacao_atual.get_estrategia(probabilidade[jogador_atual])
        proximo_jogador = (jogador_atual + 1) % self.numero_jogadores

        valores_cfr = [None] * len(Jogadas)

        for ix, jogada in enumerate(Jogadas):
            jogada_probabilidade = estrategia[ix]

           
            nova_probabilidade = probabilidade.copy()
            nova_probabilidade[jogador_atual] *= jogada_probabilidade

            
            valores_cfr[ix] = self.cfr(cartas, historico + jogada, nova_probabilidade, proximo_jogador)

        
        valores_nos = estrategia.dot(valores_cfr)
        for ix, jogada in enumerate(Jogadas):
            cfr_probabilidade = self.get_probabilidade(probabilidade, jogador_atual)
            regrets = valores_cfr[ix][jogador_atual] - valores_nos[jogador_atual]
            informacao_atual.regret_total[ix] += cfr_probabilidade * regrets
        return valores_nos

    def treinar(self, numero_iteracoes: int) -> int:
        vetor_utilidade = np.zeros(self.numero_jogadores)
        for _ in range(numero_iteracoes):
            cartas = random.sample(self.cartas, self.numero_jogadores)
            historico = ''
            probabilidade = np.ones(self.numero_jogadores)
            vetor_utilidade += self.cfr(cartas, historico, probabilidade, 0)
        return vetor_utilidade

def print_organizado(nome: str, numero_jogadores: int) -> str:
    return BARALHO[-numero_jogadores - 1:][int(nome[0])] + nome[1:]


if __name__ == "__main__":
    
    ##DEFINIR NÚMERO DE ITERAÇÕES
    numero_iteracoes = 3000000
    
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    np.random.seed(43)
    ##DEFININDO NÚMERO DE JOGADORES
    numero_jogadores = 4
    cfr_treinador = Treinador(numero_jogadores)

    vetor_utilidade = cfr_treinador.treinar(numero_iteracoes)


    for jogador in range(numero_jogadores):
        print(f"Estratégia média de cada jogador: {jogador + 1}: {(vetor_utilidade[jogador] / numero_iteracoes):.3f}")
    print("--- %s seconds ---" % (time.time() - tempo_execucao))
    print("Historico  Bet  Fold")
    for nome, informacao_atual in sorted(cfr_treinador.arvore_jogo.items(), key=lambda s: len(s[0])):
        print(f"{print_organizado(nome, numero_jogadores):3}:    {informacao_atual.get_estrategia_media()}")