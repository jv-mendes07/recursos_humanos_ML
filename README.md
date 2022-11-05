# Recursos Humanos (Modelo de Propensão)
### Projeto de Machine Learning

 Suponhamos hipoteticamente que o gerente do setor de RH (Recursos Humanos) queira que façamos uma análise descritiva e preditiva que ajude-o à prever quais são os fatores que influenciam na demissão ou na retenção de um funcionário na empresa, os objetivos do gerente ao obter tal análise são **(1)** implementar medidas no setor de RH que possam evitar a demissão futura dos funcionários e também **(2)** ter um modelo preditivo que o ajude a prever probabilisticamente a propensão de um funcionário demitir-se ou não da empresa, com base em outras variáveis influenciáveis.

Com base nos objetivos hipotéticos citados acima, realizei uma limpeza e uma análise exploratória no conjunto de dados de RH de uma empresa fictícia, e em subsequência treinei um algoritmo de regressão logística para obter um modelo preditivo que ajudasse a prever a propensão de um funcionário da empresa demitir-se ou não no futuro.

![](./img/capa.jpg)

Importei Pandas, Numpy, Seaborn e Matplotlib para realizar o processo de tratamento, manipulação e visualização de dados:

```
 import pandas as pd 
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 %matplotlib inline
  ```
Consequentemente, importei o dataset de RH com a aplicação do método .read_csv da biblioteca Pandas:

```
# Importação do conjunto de dados que será usado:

df = pd.read_csv('/content/drive/MyDrive/HR_comma.csv')
```
Após a importação e a visualização breve do dataset, avancei para a primeira etapa do projeto analítico:

### Tratamento de dados:

A primeira informação que obtive sobre o conjunto de dados usado é que há 14 mil linhas e 10 colunas contidas no dataset, além de termos 2 colunas do tipo float (número decimal), 6 do tipo int (número inteiro) e 2 do tipo object (texto).

Após a verificação dessas informações primais, realizei dois processos simples e breves de limpeza nos dados:

* Renomeação das colunas:

Mudei o formato textual do nome das colunas, para que todas colunas sejam registradas textualmente com letras minúsculas:

```
df.columns = df.columns.str.lower()
```

* Dados nulos:

Verifiquei com o método .isna().sum() para saber a quantidade de dados ausentes por coluna, felizmente foi verificável que não há nenhuma coluna com dados nulos presentes.

Concluído o processo de tratamento dos dados, comecei a focar no entendimento dos dados através da análise exploratória, para em sequência aplicar o modelo de machine learning que pudesse prever a propensão de demissão ou retenção dos funcionários da empresa.

### Análise exploratória de dados (EDA):

