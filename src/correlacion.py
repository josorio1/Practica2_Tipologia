import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

alpha = 0.05
    
def GraficosCalor(arg_data, texto):
    sns.heatmap(arg_data, annot=True, cmap='YlGnBu', vmax=1, vmin=-1)
    plt.title(f'Grafico Calor {texto}')
    plt.show()

# Carga de Datos
tintoPd = pd.read_csv('data/winequality-red.csv', delimiter=";")
blancoPd = pd.read_csv('data/winequality-white.csv', delimiter=";")

# Ignoramos la variable quality 
del(tintoPd['quality'])
del(blancoPd['quality'])

# Matriz de Correlacion Vinos Tintos
tintoCorr  = tintoPd.corr(method='pearson')
GraficosCalor(tintoCorr,  'Vinos Tintos') 

# Matriz de Correlacion Vinos Blanco
blancoCorr = blancoPd.corr(method='pearson')    
GraficosCalor(blancoCorr, 'Vinos Blancos') 
