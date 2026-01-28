# Sea4you

## Etapas do Projeto
1. Criar o setup.py: script que faz o download do dataset do kaggle --> João Henriqueq

2. AED(Análise Exploratória dos Dados) --> Laura e Maria Clara
Analisar a estrutura dos dados
Proprocessar o texto(Remover aspas, pontuações...)
Juntar legendas e imagens
Limpar os dados(Remover duplicatas e Nulos)

3. Preprocessamento do texto --> Mergulhão e Hugo
Criação da interface CustomDataset(Dataset), integrando o Dataset do PyTorch com o tipo usado para a análise exploratória
Definição da técnica de tokenização e do vocabulário considerado
Preprocessamento dos dados e criação dos DataLoaders

4. Criação do modelo --> João Henrique e Matheus
Criação da classe da Rede Convolucional
Criação da classe da Rede Recorrente
Integração de ambas as redes

5. Treinamento 
Divisão entre datasets de treino, validação e teste
Gerar gráficos da loss no decorrer das épocas(para os datasets de treino e validação) e possivelmente de outras métricas
Escolha de hiperparâmetros
Criação do script de treinamento e validação 

6. Inferência 
Executar o modelo para o dataset de teste
Extrair métricas como BLEU score, entre outras
Gerar gráficos úteis