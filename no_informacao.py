import numpy as np
#from cfr_kuhn import JOGADAS
JOGADAS = ['A', 'N']
class No_Informacao():
    #função de inicialização da classe que irá compor a árvore de jogo, o nó de informação contem o arrependimento total, a soma de todos os perfis de estratégia e o número de JOGADAS (2), check ou bet.
    def __init__(self):
        self.regret_total = np.zeros(shape=len(JOGADAS))
        self.estrategia_total = np.zeros(shape=len(JOGADAS))
        self.numero_jogadas = len(JOGADAS)

    #método de normalização para as JOGADAS serem expressas em porcentagem
    def normalizar(self, estrategia: np.array) -> np.array:
        if sum(estrategia) > 0:
            estrategia /= sum(estrategia)
        else:
            estrategia = np.array([1.0 / self.numero_jogadas] * self.numero_jogadas)
        return estrategia

    #método para a busca da estrategia a ser jogada na iteração vigente
    def get_estrategia(self, probabilidades: float) -> np.array:
        estrategia = np.maximum(0, self.regret_total)
        estrategia = self.normalizar(estrategia)
        self.estrategia_total += probabilidades * estrategia
        return estrategia

    #armazenamento do vetor estrategia total para cálculo da média
    def get_estrategia_media(self) -> np.array:
        return self.normalizar(self.estrategia_total.copy())