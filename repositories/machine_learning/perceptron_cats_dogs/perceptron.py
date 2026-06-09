import random

class Perceptron:

	# Construtor: inicializa os pesos e o bias com valores aleatórios
	def __init__(
		self,
		n_entradas,
		taxa_aprendizado=0.01
	):
		# taxa de aprendizado (eta): controla a magnitude dos ajustes nos pesos a cada iteração
		self.eta = taxa_aprendizado

		# pesos sinápticos: um por entrada, inicializados aleatoriamente no intervalo [-0.5, 0.5]
		self.pesos = [
			random.uniform(-0.5, 0.5)
			for _ in range(n_entradas)
		]

		# bias (polarização): inicializado aleatoriamente, ajusta o limiar de ativação independente das entradas
		self.bias = random.uniform(
			-0.5,
			0.5
		)

	# Calcula o potencial de ativação: soma ponderada das entradas pelos pesos, adicionando o bias
	# u = bias + (x[0]*w[0] + x[1]*w[1] + ... + x[n]*w[n])
	def soma_ponderada(self, x):
		soma = self.bias
		for i in range(len(x)):
			soma += x[i] * self.pesos[i]
		return soma

	# Função de ativação degrau (Heaviside): retorna 1 (cão) se u >= 0, ou 0 (gato) se u < 0
	def ativacao(self, valor):
		if valor >= 0:
			return 1
		return 0

	# Realiza a predição para um vetor de entrada x:
	# calcula o potencial de ativação e aplica a função degrau
	def prever(self, x):
		u = self.soma_ponderada(x)
		return self.ativacao(u)

	# Treina o Perceptron pelo número de épocas definido, ajustando pesos e bias a cada amostra
	def treinar(
		self,
		X,   # conjunto de entradas (vetores de pixels normalizados)
		y,   # rótulos esperados (0 = gato, 1 = cão)
		epocas=100
	):
		# itera pelo número de épocas de treinamento
		for epoca in range(epocas):
			erros = 0  # contador de erros na época atual

			# percorre todas as amostras do conjunto de treino
			for i in range(len(X)):
				x1 = X[i]  # vetor de pixels da amostra i
				x2 = y[i]  # rótulo esperado da amostra i

				# executa a predição para a amostra atual
				y_previsto = self.prever(x1)

				# calcula o erro: diferença entre o esperado e o previsto (E = y - ŷ)
				erro = x2 - y_previsto

				# incrementa o contador se houver erro de classificação
				if erro != 0:
					erros += 1

				# atualiza os pesos: w_j = w_j + η * E * x_j
				for j in range(len(self.pesos)):
					self.pesos[j] += (
						self.eta
						* erro
						* x1[j]
					)

				# atualiza o bias: θ = θ + η * E
				self.bias += self.eta * erro

			# imprime o numero de erros por epoca
			print(f"Época {epoca+1} Erros: {erros}")