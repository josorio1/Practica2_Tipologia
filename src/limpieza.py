import pandas as pd
import matplotlib.pyplot as plt

# Ejercicio 2 

# Cargar datasets
red_wine = pd.read_csv('data/winequality-red.csv', sep=';').drop('quality', axis=1)
white_wine = pd.read_csv('data/winequality-white.csv', sep=';').drop('quality', axis=1)

# Añadir columna 'type': toma valor 0 si el vino es tinto y 1 si es blanco 
red_wine['type'] = 0
white_wine['type'] = 1

# Combinación de los dos datasets
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

# Ejercicio 3 
# 3.1)
# Chequeo de NaNs en el dataset
has_nan = wine_data.isnull().any()

# Imputar los NaNs por la media de la variables
for column in wine_data.columns:
    if has_nan[column]:
        print(f"La columna {column} tiene NaNs. Imputaremos ests valores tomando la media de la columna")
        wine_data[column].fillna(wine_data[column].mean(), inplace=True)
if sum(has_nan) == 0:
    print("Ninguna columna tienen NaNs")
    
# 3.2)
# Columnas a estudiar
columns_to_plot = [col for col in wine_data.columns if col != 'type']
# Preparación de los boxplots
num_plots = len(columns_to_plot)
num_cols = 3 
num_rows = (num_plots + num_cols - 1) // num_cols  #
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8)) 
axes = axes.flatten()
for i, column in enumerate(columns_to_plot):
    axes[i].boxplot(wine_data[column])
    axes[i].set_ylabel(column)
    axes[i].tick_params(axis='x', labelsize=0.1)
if num_plots < num_cols * num_rows:
    for j in range(num_plots, num_cols * num_rows):
        fig.delaxes(axes[j])
fig.tight_layout()
plt.show()



# Generación dataset definitivo
wine_data.to_csv("data/winequality_combinado.csv", index=False)

