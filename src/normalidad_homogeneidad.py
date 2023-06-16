import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

alpha = 0.05

def GraficosQQ (arg_data, texto):
    fig, ax = plt.subplots(figsize=(7,4))
    sm.qqplot(arg_data, fit = True, line = 'q', alpha = 0.4, lw = 2, ax = ax)
    ax.set_title(texto, fontsize = 10, fontweight = "bold")
    ax.tick_params(labelsize = 7)
    plt.show()
    

def normalizacion(arg_data, vino, texto):

    kur     = stats.kurtosis(arg_data)
    skew    = stats.skew(arg_data)
    shapiro = stats.shapiro(arg_data).pvalue

    if vino == 'Blanco':
        blancoNorm.at[texto, 'kurtosis']       = kur
        blancoNorm.at[texto, 'pasa-kurtosis']  = (False, True)[kur == 0]         
        blancoNorm.at[texto, 'skewness']       = skew
        blancoNorm.at[texto, 'pasa-skewness']  = (False, True)[skew == 0]          
        blancoNorm.at[texto, 'shapiro']        = shapiro
        blancoNorm.at[texto, 'pasa-shapiro']   = (False, True)[(shapiro>=alpha)]
    else:
        tintoNorm.at[texto, 'kurtosis']        = kur
        tintoNorm.at[texto, 'pasa-kurtosis']   = (False, True)[kur == 0]          
        tintoNorm.at[texto, 'skewness']        = skew
        tintoNorm.at[texto, 'pasa-skewness']   = (False, True)[skew == 0]        
        tintoNorm.at[texto, 'shapiro']         = shapiro
        tintoNorm.at[texto, 'pasa-shapiro']    = (False, True)[(shapiro>=alpha)]


# Carga de Datos
tintoPd = pd.read_csv('data/winequality-red.csv', delimiter=";")
blancoPd = pd.read_csv('data/winequality-white.csv', delimiter=";")

# Ignoramos la variable quality 
del(tintoPd['quality'])
del(blancoPd['quality'])

# Creamos DataSet de Normalizacion
tintoNorm  = pd.DataFrame(columns=['kurtosis','pasa-kurtosis','skewness','pasa-skewness','shapiro','pasa-shapiro'], index=tintoPd.columns)
blancoNorm = pd.DataFrame(columns=['kurtosis','pasa-kurtosis','skewness','pasa-skewness','shapiro','pasa-shapiro'], index=blancoPd.columns)

# Creamos DataSet de Test (homocedasticidad)
varianzasTest = pd.DataFrame(columns=['levene','pasa-levene', 'bartlett','pasa-bartlett', 'fligner','pasa-fligner'], index=tintoPd.columns)


# Analizamos cada una de las variables
for columna in tintoPd.columns:
    
    # Normalizacion
    normalizacion(tintoPd[columna], 'Tinto', columna)
    normalizacion(blancoPd[columna], 'Blanco', columna)

    # levene
    t_stat, p_value = stats.levene(tintoPd[columna], blancoPd[columna], center='median')
    varianzasTest.at[columna, 'levene']      = p_value
    varianzasTest.at[columna, 'pasa-levene'] = (False, True)[p_value>=alpha]
 
    # bartlett
    t_stat, p_value = stats.bartlett(tintoPd[columna], blancoPd[columna])
    varianzasTest.at[columna, 'bartlett']      = p_value
    varianzasTest.at[columna, 'pasa-bartlett'] = (False, True)[p_value>=alpha]

    # fligner
    t_stat, p_value = stats.fligner(tintoPd[columna], blancoPd[columna], center='median')
    varianzasTest.at[columna, 'fligner']      = p_value
    varianzasTest.at[columna, 'pasa-fligner'] = (False, True)[p_value>=alpha]
    
    # Gráfico Q-Q
    GraficosQQ(blancoPd[columna], f'Vino Blanco - {columna}')
    GraficosQQ(tintoPd[columna], f'Vino Tinto - {columna}')

# Mostramos los resultados    
print(tintoNorm)
print(blancoNorm)
print(varianzasTest)

