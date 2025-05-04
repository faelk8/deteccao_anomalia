import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from prophet import Prophet
from datetime import datetime, timedelta
import random


def gerar_dados_falsos():
    """
    Gera dados falsos para treino de modelo de anomalias.

    Gera dados de pedidos por hora, com sazonalidade:
        - manh : 20% a mais;
        - fds : 30% a mais;
        - 1, 15, 30 do ms : 50% a mais;
    Com anomalias negativas (quedas) em 0.2% dos registros.

    Retorna um DataFrame com colunas 'data', 'origem' e 'quantidade'.
    """
    datas = pd.date_range(start='2024-01-01', end='2025-03-31', freq='H')
    origens = ['site', 'marketplace', 'manual']

    dados = []

    for data in datas:
        for origem in origens:
            base = 100 if origem == 'site' else 80 if origem == 'marketplace' else 30
            hora = data.hour
            dia_semana = data.dayofweek
            dia_mes = data.day

            # Sazonalidade
            if hora in range(9, 18):
                base *= 1.2
            if dia_semana in [5, 6]:
                base *= 1.3
            if dia_mes in [1, 15, 30]:
                base *= 1.5

            quantidade = np.random.poisson(base)

            # Injetar quedas (anomalias negativas)
            if random.random() < 0.002:
                quantidade = max(0, int(quantidade * random.uniform(0.1, 0.4)))

            dados.append({'data': data, 'origem': origem,
                         'quantidade': quantidade})

    return pd.DataFrame(dados)


def detectar_anomalias(df):
    """
    Detecta anomalias em uma série temporal de pedidos.

    Utiliza Z-Score e Isolation Forest para detectar anomalias. Além disso, 
    utiliza o Prophet apenas para a origem "site".

    Retorna um DataFrame com as colunas 'data', 'origem', 'quantidade',
    'anomalia_z', 'anomalia_iso' e 'anomalia_prophet'.
    """
    df = df.copy()
    df['data'] = pd.to_datetime(df['data'])
    df['hora'] = df['data'].dt.hour
    df['dia_semana'] = df['data'].dt.dayofweek
    df['dia_mes'] = df['data'].dt.day

    # Z-Score
    stats = df.groupby(['origem', 'hora', 'dia_semana'])[
        'quantidade'].agg(['mean', 'std']).reset_index()
    df = df.merge(stats, on=['origem', 'hora', 'dia_semana'], how='left')
    df['z_score'] = (df['quantidade'] - df['mean']) / df['std']
    df['anomalia_z'] = df['z_score'] < -2.5

    # Isolation Forest
    iso_model = IsolationForest(contamination=0.01, random_state=42)
    df['anomalia_iso'] = iso_model.fit_predict(
        df[['quantidade', 'hora', 'dia_semana', 'dia_mes']])
    df['anomalia_iso'] = df['anomalia_iso'] == -1

    # Prophet (aplicado apenas na origem "site")
    df_prophet = df[df['origem'] == 'site'][['data', 'quantidade']].rename(
        columns={'data': 'ds', 'quantidade': 'y'})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(df_prophet)
    forecast = model.predict(df_prophet)
    df.loc[df['origem'] == 'site', 'yhat'] = forecast['yhat'].values
    df.loc[df['origem'] == 'site', 'yhat_lower'] = forecast['yhat_lower'].values
    df.loc[df['origem'] == 'site', 'anomalia_prophet'] = df.loc[df['origem']
                                                                == 'site', 'quantidade'] < df.loc[df['origem'] == 'site', 'yhat_lower']

    return df


def plot_anomalias(df, origem='site', metodo='z'):
    """
    Plota gráfico com anomalias detectadas por Z-Score, Isolation Forest ou Prophet.

    Parâmetros:
        df (pandas.DataFrame): DataFrame com colunas 'data', 'origem' e 'quantidade'.
        origem (str): Origem dos pedidos ('site', 'marketplace', 'manual'). Padrão é 'site'.
        metodo (str): Método de detecção de anomalias ('z', 'iso', 'prophet'). Padrão é 'z'.

    Retorna:
        None
    """
    df_origem = df[df['origem'] == origem].sort_values(by='data')

    if metodo == 'z':
        df_origem['anomaly'] = df_origem['anomalia_z']
        title = 'Z-Score'
    elif metodo == 'iso':
        df_origem['anomaly'] = df_origem['anomalia_iso']
        title = 'Isolation Forest'
    elif metodo == 'prophet':
        df_origem['anomaly'] = df_origem['anomalia_prophet']
        title = 'Prophet'
    else:
        raise ValueError("Método deve ser: 'z', 'iso' ou 'prophet'.")

    plt.figure(figsize=(14, 6))
    plt.plot(df_origem['data'], df_origem['quantidade'],
             label='Pedidos', alpha=0.6)
    plt.scatter(df_origem[df_origem['anomaly']]['data'], df_origem[df_origem['anomaly']]['quantidade'],
                color='red', label='Anomalia', zorder=5)
    plt.title(f'Anomalias Detectadas - {title} - Origem: {origem}')
    plt.xlabel('Data')
    plt.ylabel('Quantidade de Pedidos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = gerar_dados_falsos()
    df_anomalias = detectar_anomalias(df)

    # Exibir gráficos
    plot_anomalias(df_anomalias, origem='site', metodo='z')
    plot_anomalias(df_anomalias, origem='site', metodo='iso')
    plot_anomalias(df_anomalias, origem='site', metodo='prophet')
