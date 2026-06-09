from PIL import Image
import numpy as np
import pandas as pd
import os

# Processa o dataset, embaralha e divide em treino e teste
def processar_e_dividir_dataset(pasta_base, proporcao_treino=0.8):
	dados = []

	# Define as classes para classificação
	classes = {
		"cat": 0,
		"dog": 1
	}

	for pasta, classe in classes.items():
		# caminho a percorrer
		caminho = os.path.join(pasta_base, pasta)

		# percorre os diretórios
		for arquivo in os.listdir(caminho):
			# pega apenas aquivos validos 
			if not arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
				continue

			# abre o arquivo
			imagem = Image.open(os.path.join(caminho, arquivo))

			# converte para escala de cinza
			imagem = imagem.convert("L")

			# redimensiona imagem
			imagem = imagem.resize((32, 32))

			# transforma em array
			pixels = np.array(imagem)

			# faz o achatamento da imagem, convertendo uma matriz bidimensional (2D) em um vetor unidimensional (1D)
			vetor = pixels.flatten()

			# ormalização dos dados por meio do redimensionamento de escala (conhecido como Min-Max Scaling)
			# ao dividir cada elemento do vetor por 255.0, o intervalo original de [0, 255] é mapeado proporcionalmente para um intervalo contínuo de [0.0, 1.0]
			vetor = vetor / 255.0

			# transforma em uma lista
			linha = vetor.tolist()

			# adciona a linha da classe
			linha.append(classe)

			# adiciona do dicionario
			dados.append(linha)

	# gera o dataframe 
	df = pd.DataFrame(dados)

	# embaralha as linhas de forma aleatória para garantir distribuição uniforme de classes
	df_embaralhado = df.sample(frac=1, random_state=42).reset_index(drop=True)

	# calcula o ponto de divisão (80% para treino, 20% para teste)
	limite_treino = int(len(df_embaralhado) * proporcao_treino)

	# limite sequindo a proporção
	df_treino = df_embaralhado.iloc[:limite_treino]
	df_teste = df_embaralhado.iloc[limite_treino:]

	# pasta de saída
	pasta_saida = 'dataset_normalizado'
	os.makedirs(pasta_saida, exist_ok=True)

	# exporta treino para csv
	caminho_treino = os.path.join(pasta_saida, 'treino.csv')
	df_treino.to_csv(
		caminho_treino,
		index=False,		# sem index nas linhas
		header=False		# sem cabeçalho
	)

	# exporta teste para csv
	caminho_teste = os.path.join(pasta_saida, 'teste.csv')
	df_teste.to_csv(
		caminho_teste,
		index=False,		# sem index nas linhas
		header=False		# sem cabeçalho
	)

	print(
		f"CSV de Treino gerado: {caminho_treino} ({len(df_treino)} amostras)"
	)
	print(
		f"CSV de Teste gerado: {caminho_teste} ({len(df_teste)} amostras)"
	)

if __name__ == "__main__":
	# Processo de normalização e divisão (80% treino, 20% teste)
	processar_e_dividir_dataset("dataset", proporcao_treino=0.8)





