import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

alpha = 0.05
# Carga de Datos
tintoPd = pd.read_csv('data/winequality-red.csv', delimiter=";")
blancoPd = pd.read_csv('data/winequality-white.csv', delimiter=";")

# Ignoramos la variable quality 
del(tintoPd['quality'])
del(blancoPd['quality'])

# Creamos DataSet de Test (t-test)
vinosTest = pd.DataFrame(columns=['t-test','pasa-test'], index=tintoPd.columns)

# Analizamos cada una de las variables
for columna in tintoPd.columns: 
    # T test
    t_stat, p_value = stats.ttest_ind(tintoPd[columna], blancoPd[columna])
    vinosTest.at[columna, 't-test']      = p_value
    vinosTest.at[columna, 'pasa-test']   = (False, True)[p_value>=alpha]
    
# Mostramos los resultados del t-test    
print(vinosTest)

