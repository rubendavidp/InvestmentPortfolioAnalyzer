import pandas as pd
import pandas_datareader as wb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.optimize as optimize


#--- Acciones a analizar 

tickers = ['TSLA', 'AAPL', 'DASH']
start_date = '2020-01-01'
end_date = '2022-01-31'


#--- Dataframe de precios

data = pd.DataFrame(wb.DataReader(tickers, 'yahoo', start_date, end_date)['Close'])
returns = (data.pct_change()*100)
corr = data.corr()


#--- MARKOWITZ

port_returns = []
port_vols = []

for i in range (100):
    num_assets = len(tickers)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    ret_esp = np.sum(returns.mean() * weights)    ##--- Retorno esperado del portafolio
    var_esp = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))  ##--- Volatilidad esperada del portafolio
    port_returns.append(ret_esp)
    port_vols.append(var_esp)

## --- Markowitz's bullets
plt.figure(figsize = (12,6))
plt.scatter(port_vols, port_returns)
plt.xlabel('Volatilidad de la cartera')
plt.ylabel('Retorno de la cartera')
plt.show()



"""print('\nMatriz de covarianza:\n',returns.cov())
print('\nEl retorno promedio de las acciones es:\n\n', returns.mean())
print('\n\nEl precio promedio de las acciones es:\n\n', data.mean())
print('\n\nMatriz de correlación:\n\n', corr)"""



#--- GRAPHS
#  
#Gráfica de evolución de los precios (relativa)
"""(data / data.iloc[0]*100).plot()
plt.show()

#Histograma
sns.histplot(df)
plt.xlabel('retornos diarios')
plt.ylabel('frecuencia')
plt.show()

#Heatmap de la matriz de correlación
sns.heatmap(corr, annot = True, linewidths = 1)
plt.title("Matriz de correlación")
plt.show()"""