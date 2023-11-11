class Galinha:
    def __init__(self, galinha_id, pos_x, pos_y):
        self.galinha_id = galinha_id
        self.distancia_percorrida = 0
        self.tempo_parado = 0
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.tempo_comendo = 0
        self.tempo_bebendo = 0

    def atualizar_posicao(self, new_pos_x, new_pos_y):
        # Calcula a distância percorrida desde a última atualização
        distancia = ((new_pos_x - self.pos_x) ** 2 + (new_pos_y - self.pos_y) ** 2) ** 0.5
        self.distancia_percorrida += distancia

        # Atualiza a posição atual
        self.pos_x = new_pos_x
        self.pos_y = new_pos_y

    def incrementar_tempo_parado(self):
        self.tempo_parado += 1
