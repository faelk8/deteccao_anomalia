<h1 align="center">📊 Detecção de Anomalias em Pedidos com Sazonalidade</h1>

Este projeto simula e detecta anomalias em um conjunto de dados de pedidos com variações sazonais (hora do dia, dia da semana, dia do mês). O foco é identificar quedas anormais nos pedidos, utilizando diferentes abordagens estatísticas e de aprendizado de máquina.

# 🔧 Tecnologias utilizadas
Python 3.10+

pandas, numpy

matplotlib

scikit-learn

prophet (Facebook Prophet)


# 🎯 Objetivo
Detectar anomalias negativas (quedas atípicas na quantidade de pedidos) levando em consideração:

Origem do pedido (site, marketplace)

Hora do dia

Dia da semana

Dia do mês


# 🧪 Etapas do Projeto
Geração de Dados Sintéticos

Simula dados horários de pedidos por origem.

Incorpora padrões sazonais reais:

Mais pedidos em horários comerciais.

Mais pedidos aos finais de semana.

Dias específicos do mês com maior volume (ex: dia 1, 15 e 30).

Injeta quedas artificiais (anômalas) com baixa probabilidade.

Modelos de Detecção de Anomalias

# 📉 1. Z-Score (Estatística)
Cálculo do desvio padrão e média por hora/dia da semana/origem.

Considera anômalo se o Z-Score for muito abaixo da média (ex: < -2.5).

✅ Simples e eficaz para séries com padrão conhecido.

# 🌲 2. Isolation Forest (Machine Learning)
Algoritmo baseado em árvores que "isola" pontos anômalos.

Detecta outliers multivariados com base em:

quantidade, hora, dia da semana e dia do mês.

✅ Não supervisionado, bom para conjuntos complexos.

# 🔮 3. Prophet (Facebook)
Modelo de séries temporais que lida com sazonalidade automaticamente.

Treinado para prever a quantidade esperada e limites de confiança.

Um ponto é anômalo se estiver abaixo do limite inferior previsto.

✅ Ideal para tendências e sazonalidades fortes.

# Visualização

Utiliza matplotlib para exibir gráficos de:

Série de pedidos ao longo do tempo.

Anomalias detectadas para cada método.

Visualizações separadas por origem (foco no "site" como exemplo).

