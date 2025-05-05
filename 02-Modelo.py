import pandas as pd
import numpy as np

from scipy.stats import zscore
# 1. Gerar dados simulados a cada 30 segundos
np.random.seed(42)
date_range = pd.date_range(start="2024-01-01 00:00:00",
                           end="2025-01-01 00:00:00", freq="1S")

origens = ['site', 'marketplace']
data = []

for dt in date_range:
    for origem in origens:
        qtd = np.random.poisson(lam=3)  # Média de 3 pedidos a cada 30s
        data.append([dt, qtd, origem])

df = pd.DataFrame(data, columns=["ped_data_hora", "quantidade", "ped_origem"])

# 2. Agrupar por 5 minutos
df["ped_data_hora"] = pd.to_datetime(df["ped_data_hora"])
df_agrupado = (
    df
    .set_index("ped_data_hora")
    .groupby([pd.Grouper(freq="5min"), "ped_origem"])
    .sum()
    .reset_index()
)


# Aplica z-score dentro de cada origem
def detectar_anomalias_zscore(df, threshold=2.5):
    df_resultado = df.copy()
    df_resultado["z_score"] = (
        df_resultado
        .groupby("ped_origem")["quantidade"]
        .transform(zscore)
    )
    df_resultado["anomalous"] = df_resultado["z_score"].abs() > threshold
    return df_resultado


# Aplicar a função
df_com_anomalias = detectar_anomalias_zscore(df_agrupado, threshold=2.5)

# Mostrar exemplos de anomalias detectadas
print(df_com_anomalias[df_com_anomalias["anomalous"]
                       ].sort_values("ped_data_hora").head())
