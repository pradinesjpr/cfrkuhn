from typing import List, Dict
import random
import numpy as np
import time
from jogo import RegrasDeJogo
from no_informacao import No_Informacao
tempo_execucao = time.time()
JOGADAS = ['A', 'N']  # apostar/completar apostar, não apostar(desistir)/ check
BARALHO = ['9', 'D', 'J', 'Q', 'K'] # baralho, na versão de 2 jogadores somente são utilizados k,q,j

class Treinador_CFR():
    def __init__(self, numero_jogadores: int):
        self.arvore_jogo: Dict[str, No_Informacao] = {}
        self.numero_jogadores = numero_jogadores
        self.cartas = [x for x in range(self.numero_jogadores + 1)]

    #algoritmo vai se situar em que parte da árvore de jogo ele está utilizando a string carta + histórico que seria algo do tipo J aa (jogador 1 possui J e aposta, jogador 2 aposta também)
    def get_informacao_no(self, carta_e_historico: str) -> No_Informacao:
        if carta_e_historico not in self.arvore_jogo:
            self.arvore_jogo[carta_e_historico] = No_Informacao()
        return self.arvore_jogo[carta_e_historico]


    #define a probabilidades de se alcançar aquele nó da história do jogo diante das chances de se escolher check/bet de cada jogador. produto de probabilidades de se alcançar um evento

    def get_probabilidades_acumuladas(self, probabilidades: np.array, jogador: int):
        return np.prod(probabilidades[:jogador]) * np.prod(probabilidades[jogador + 1:])


    #metodo principal do algoritmo counterfactural regret minimization
    def cfr(self, cartas: List[str], historico: str, probabilidades: np.array, jogador_atual: int) -> np.array:
        
        #verifica se o jogo acabou
        if RegrasDeJogo.is_estado_terminal(historico, self.numero_jogadores):
            return RegrasDeJogo.get_ganho(cartas, historico, self.numero_jogadores)

        minha_carta = cartas[jogador_atual]
        no_atual = self.get_informacao_no(str(minha_carta) + historico)
        estrategia = no_atual.get_estrategia(probabilidades[jogador_atual])
        proximo_jogador = (jogador_atual + 1) % self.numero_jogadores



        valores_cfr = [None] * len(JOGADAS)

        for ix, jogada in enumerate(JOGADAS):
            jogada_probabilidade = estrategia[ix]
            nova_probabilidade = probabilidades.copy()
            nova_probabilidade[jogador_atual] *= jogada_probabilidade
            valores_cfr[ix] = self.cfr(cartas, historico + jogada, nova_probabilidade, proximo_jogador)
        valores_nos = estrategia.dot(valores_cfr)
        for ix, jogada in enumerate(JOGADAS):
            cfr_probabilidade = self.get_probabilidades_acumuladas(probabilidades, jogador_atual)
            regrets = valores_cfr[ix][jogador_atual] - valores_nos[jogador_atual]
            no_atual.regret_total[ix] += cfr_probabilidade * regrets
        return valores_nos

    def treinar(self, numero_iteracoes: int) -> int:
        vetor_utilidade = np.zeros(self.numero_jogadores)
        for _ in range(numero_iteracoes):
            cartas = random.sample(self.cartas, self.numero_jogadores)
            historico = ''
            probabilidades = np.ones(self.numero_jogadores)
            vetor_utilidade += self.cfr(cartas, historico, probabilidades, 0)
        
        return vetor_utilidade

def print_organizado(nome: str, numero_jogadores: int) -> str:
    return BARALHO[-numero_jogadores - 1:][int(nome[0])] + nome[1:]


if __name__ == "__main__":
    
    ##DEFINIR NÚMERO DE ITERAÇÕES DE MONTE CARLO
    numero_iteracoes = 50000
    
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)
    np.random.seed(43)
    ##DEFININDO NÚMERO DE JOGADORES
    numero_jogadores = 2
    cfr_treinador = Treinador_CFR(numero_jogadores)

    vetor_utilidade = cfr_treinador.treinar(numero_iteracoes)
    print(f"Rodando por {numero_iteracoes} iterações e {numero_jogadores} jogadores.")
    for jogador in range(numero_jogadores):
        print(f"Estratégia média de cada jogador {jogador + 1}: {(vetor_utilidade[jogador] / numero_iteracoes):.3f}")
    print("--- %s segundos ---" % (time.time() - tempo_execucao))
    print("Hist    A     N")
    for nome, no_atual in sorted(cfr_treinador.arvore_jogo.items(), key=lambda s: len(s[0])):
        print(f"{print_organizado(nome, numero_jogadores):3}:    {no_atual.get_estrategia_media()}")