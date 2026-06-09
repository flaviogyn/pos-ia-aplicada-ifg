import csv
from perceptron import Perceptron

# Função para carregar os dados do arquivo CSV
def carregar_csv(nome_arquivo):
	X = []  # Armazena as características da imagem (vetor de pixels)
	y = []  # Armazena a classe/rótulo (0 para gato, 1 para cão)

	# abrindo arquivo
	with open(
		nome_arquivo,
		"r"
	) as arquivo:

		# lendo o arquivo csv
		leitor = csv.reader(arquivo)

		for linha in leitor:
			# Converte os valores da linha de string para float
			linha = [
				float(x)
				for x in linha
			]

			# As características são todos os elementos, exceto a última coluna
			X.append(
				linha[:-1]
			)

			# A classe (alvo) é o último elemento da linha
			y.append(
				int(linha[-1])
			)

	return X, y

print("Carregando treino...")

# Carrega as amostras e rótulos de treinamento
X_treino, y_treino = carregar_csv(
  "dataset_normalizado/treino.csv"
)

print("Carregando teste...")
# Carrega as amostras e rótulos de teste (dados inéditos)
X_teste, y_teste = carregar_csv(
  "dataset_normalizado/teste.csv"
)

# Determina o número de entradas do Perceptron (tamanho do vetor de pixels, ex: 1024)
n_entradas = len(
  X_treino[0]
)

print(
  "Número de entradas:",
  n_entradas
)

# Cria a instância do Perceptron com a taxa de aprendizado definida
rede = Perceptron(
	n_entradas=n_entradas,
	taxa_aprendizado=0.01
)

print("Treinando...")

# Treina o modelo utilizando o conjunto de treino ao longo de 100 épocas
rede.treinar(
	X_treino,
	y_treino,
	epocas=100
)

print("Testando...")

# Avalia a capacidade de generalização do modelo no conjunto de teste
acertos = 0

for i in range(len(X_teste)):
	# Executa a predição para a amostra de teste atual
	previsto = rede.prever(
		X_teste[i]
	)

	esperado = y_teste[i]

	# Verifica se a predição coincide com o rótulo esperado
	if previsto == esperado:
		acertos += 1

# Total de amostras testadas
total = len(
	X_teste
)

# Calcula a acurácia (porcentagem de acertos sobre o total)
acuracia = (
	acertos / total
) * 100

# Imprime os resultados finais de avaliação
print()
print(f"Acertos: {acertos}")
print(f"Erros: {total - acertos}")
print(f"Total: {total}")
print(f"Acurácia: {acuracia:.2f}%")