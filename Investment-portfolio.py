import pandas as pd
import pandas_datareader as wb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.optimize as optimize

#-- Acciones a analizar 
tickers = []
num_assets = int(input('Número de acciones a analizar:\n'))

for i in range(num_assets):
    tickers.append(input('Acción a analizar:\n').upper())

start_date = input('Fecha de inicio (YYYY-MM-DD):\n')
end_date = input('Fecha final (YYYY-MM-DD):\n')
data = pd.DataFrame(wb.DataReader(tickers, 'yahoo', start_date, end_date)['Close'])
returns = (data.pct_change()*100)
corr = data.corr()

#-- Dataframe de precios de cierre

print('\nMatriz de covarianza:\n', returns.cov())
print('\nEl retorno promedio de las acciones es:\n\n', returns.mean())
print('\n\nEl precio promedio de las acciones es:\n\n', data.mean())
print('\n\nMatriz de correlación:\n\n', corr)

#-- Markowitz
port_returns = []
port_vols = []

for i in range (10000):
    num_assets = len(tickers)
    weights = np.random.random(num_assets)
    weights /= np.sum(weights) 
    ret_esp = np.sum(returns.mean() * weights) #--> Retorno esperado del portafolio
    var_esp = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights))) #--> Volatilidad esperada del portafolio
    port_returns.append(ret_esp)
    port_vols.append(var_esp)

def portfolio_stats(weights, log_returns):
    ret_esp = np.sum(log_returns.mean() * weights)
    var_esp = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov(), weights)))
    sharpe = ret_esp/var_esp    #--> Agregar el % de riskfree
    return {'Return': ret_esp, 'Volatility': var_esp, 'Sharpe': sharpe}
#-- Minimización del sharpe negativo = Maximización
def minimize_sharpe(weights, log_returns): 
    return -portfolio_stats(weights, log_returns)['Sharpe'] 

port_returns = np.array(port_returns)
port_vols = np.array(port_vols)
sharpe = port_returns/port_vols

max_sr_vol = port_vols[sharpe.argmax()]
max_sr_ret = port_returns[sharpe.argmax()]

constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1}) #--> Restricción de que todas las ponderaciones de la cartera deben sumar 1
bounds = tuple((0,1) for x in range(num_assets)) #--> % que puede ocupar un activo en la cartera
initializer = num_assets * [1./num_assets,] #--> Cada acción ocupará el mismo % en la cartera

optimal_sharpe = optimize.minimize(minimize_sharpe, initializer, method = 'SLSQP', args = (returns,) ,bounds = bounds, constraints = constraints)
optimal_sharpe_weights = optimal_sharpe['x'].round(4)
optimal_stats = portfolio_stats(optimal_sharpe_weights, returns)

print('\n\nPesos óptimos de la cartera: ', list(zip(tickers, list(optimal_sharpe_weights*100))))
print('\n\nRetorno óptimo de la cartera: ', round(optimal_stats['Return']*100,4))
print('\n\nVolatilidad óptima de la cartera: ', round(optimal_stats['Volatility']*100,4))
print('\n\nRatio Sharpe óptimo de la cartera: ', round(optimal_stats['Sharpe'],4))

#-- Markowitz's Graph
plt.figure(figsize = (12,6))
plt.scatter(port_vols,port_returns,c = (port_returns/port_vols))
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=30)
plt.colorbar(label = 'Ratio Sharpe (rf=0)')
plt.xlabel('Volatilidad de la cartera')
plt.ylabel('Retorno de la cartera')
plt.show()


#-- Graphs 
#Gráfica de evolución de los precios (relativa)
(data / data.iloc[0]*100).plot()
plt.show()

#-- Histograma
sns.histplot(returns)
plt.xlabel('Retornos diarios')
plt.ylabel('Frecuencia')
plt.show()

#-- Heatmap de la matriz de correlación
sns.heatmap(corr, annot=True, linewidths=1)
plt.title("Matriz de correlación")
plt.show()
