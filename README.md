# Recursos Humanos (Modelo de Propensão)
### Projeto de Machine Learning

 Suponhamos hipoteticamente que o gerente do setor de RH (Recursos Humanos) queira que façamos uma análise descritiva e preditiva que ajude-o à prever quais são os fatores que influenciam na demissão ou na retenção de um funcionário na empresa, os objetivos do gerente ao obter tal análise são **(1)** implementar medidas no setor de RH que possam evitar a demissão futura dos funcionários e também **(2)** ter um modelo preditivo que o ajude a prever probabilisticamente a propensão de um funcionário demitir-se ou não da empresa, com base em outras variáveis influenciáveis.

Com base nos objetivos hipotéticos citados acima, realizei uma limpeza e uma análise exploratória no conjunto de dados de RH de uma empresa fictícia, e em subsequência treinei um algoritmo de regressão logística para obter um modelo preditivo que ajudasse a prever a propensão de um funcionário da empresa demitir-se ou não no futuro.

![](./img/capa.jpg)

Para iniciar o projeto, importei as bibliotecas Pandas, Numpy, Seaborn e Matplotlib para realizar o processo de tratamento, manipulação e visualização de dados:

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

Mudei o formato textual do nome das colunas, para que todas colunas fossem registradas textualmente com letras minúsculas:

```
df.columns = df.columns.str.lower()
```

* Dados nulos:

Apliquei o método .isna().sum() para saber a quantidade de dados ausentes por coluna, felizmente foi verificável que não há nenhuma coluna com dados nulos presentes.

Concluído o processo de tratamento dos dados, comecei a focar no entendimento dos dados através da análise exploratória, para em sequência aplicar o modelo de machine learning que pudesse prever a propensão de demissão ou retenção dos funcionários da empresa.

### Análise exploratória de dados (EDA):

Antes de iniciar a etapa de análise exploratória, é indispensável saber sobre o quê cada coluna se trata:

* satisfaction_level: Taxa de satisfação do funcionário com a empresa (entre a faixa de 0 e 1).
* last_evaluation: Última avaliação de desempenho que os funcionários receberam (entre a faixa de 0 e 1).
* number_project: Número de projetos que os funcionários participaram na empresa.
* average_montly_hours: Média de horas de trabalho que cada funcionário gastou na empresa mensalmente.
* time_spend_company: Anos de trabalho que cada funcionário teve na empresa.
* work_acident: Informação sobre acidente de trabalho sofrido por cada funcionário (0 para não e 1 para sim).
* left: Informação sobre o fato de cada funcionário ter sido demitido ou não da empresa (0 para não e 1 para sim).
* promotion_last_5_years: Promoção na empresa nos últimos 5 anos para cada funcionário (0 para não e 1 para sim).
* department: Departamento que cada funcionário trabalha na empresa.
* salary: Classificação salarial de cada funcionário entre baixo, médio e alto salário.

Após ter explicitado o quê cada variável significa, comecei tal análise com a seguinte questão:

**(1)** **Qual é a quantidade de funcionários demitidos e retidos da empresa?**

Com essa questão, quis saber a quantidade de funcionários que foram demitidos e a quantidade de funcionários que continuam na empresa, e como resposta obtive que 11 mil funcionários continuam na empresa e outros 3 mil funcionários foram demitidos da empresa, percentualmente expus tal informação com um gráfico de pizza:

![](./img/gr_1.png)

Notavelmente, observamos que mais que 3 / 4 dos funcionários continuam na empresa e somente 23 % dos funcionários foram demitidos.

À partir dessa informação inicial, quis saber a quantidade de funcionário por classificação salarial, para depois obter mais insights sobre o dataset:

**(2)** **Qual é a porcentagem de funcionários por classificação salarial?**

Com um gráfico de rosca expus a resposta informacional para a questão acima:

![](./img/gr_2.png)

Como é observável acima, a suma maioria de 48 % dos funcionários recebem um salário considerado baixo, 43 % recebem um salário consideravelmente médio, e por fim uma minoria ínfima de funcionários recebem um salário considerado alto.

Após saber a quantidade de funcionários demitidos e retidos, e após saber a quantidade de funcionários por classificação salarial, foi necessário saber a relação entre a taxa de demissão e retenção dos funcionários por classificação salarial:

**(3)** **Qual é a taxa de retenção e demissão dos funcionários por classificação salarial?**

Antes de responder tal questão com um gráfico, manipulei os dados para obter uma tabela que trouxesse a quantidade e a porcentagem de funcionários demitidos e retidos por classificação salarial:


|        |      | qtd_left | perc_left |
|--------|------|----------|-----------|
| salary | left |          |           |
| high   | 0    | 1155     | 7.70      |
|        | 1    | 82       | 0.55      |
| low    | 0    | 5144     | 34.30     |
|        | 1    | 2172     | 14.48     |
| medium | 0    | 5129     | 34.20     |
|        | 1    | 1317     | 8.78      |

Com os dados da tabela acima, expus a quantidade de funcionários retidos ou demitidos por classificação salarial através de um gráfico de barras que expusesse intuitivamente tais informações:

![](./img/gr_3.png)

O gráfico de barras acima fornece os seguintes insights:

* Há uma quantidade aproximadamente equivalente de funcionários que continuam na empresa e recebem salários baixos ou médios.
* Os funcionários que foram demitidos majoritariamente recebiam salários considerados baixo.
* Os funcionários que recebem altos salários em suma maioria continuam na empresa.

Após tais informações, poderíamos questionar **(a)** se baixos salários influenciam na demissão dos funcionários ou **(b)** se altos salários influenciam na retenção na retenção dos funcionários, no gráfico acima consegui algumas evidências que podem confirmar a tendência de tais hipóteses.

Para continuar a análise com mais aprofundamento, explorei a quantidade de funcionários demitidos e retidos por departamento, para ver se há alguma relação entre retenção e demissão com o departamento que o funcionário trabalha:

**(4)** Qual é a quantidade de retenções e demissões de funcionários por departamento?

