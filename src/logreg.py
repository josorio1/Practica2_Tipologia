import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
import itertools
import numpy as np


# Combinación de los dos datasets
wine_data = pd.read_csv("data/winequality_combinado.csv")

# Variables dependientes
cols = [col for col in wine_data.columns if col != 'type']
best_aic = 10e12  # Initialize with a large value
best_model = None

# División entre conjunto de entrenamiento y test
x_data = wine_data[cols]  # Replace "target_column_name" with the actual name of the target column
y_data = wine_data["type"]  # Replace "target_column_name" with the actual name of the target column

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# Iterate through all possible combinations of features and select the model with the lowest AIC
for k in range(1, len(cols) + 1):
    for subset in set(itertools.combinations(cols, k)):
        X = X_train[list(subset)]
        X = sm.add_constant(X)  # Add constant term for intercept
        y = y_train
        model = sm.Logit(y, X)
        result = model.fit()
        if result.aic < best_aic:
            best_aic = result.aic
            best_model = result
            best_subset = subset

print(f"El mejor AIC es:", best_aic)
print(f"El mejor modelo es:\n", best_model.summary())


# Generar prediciones
X_test_best = sm.add_constant(X_test[list(best_subset)])
predictions = best_model.predict(X_test_best)

# Sacar métricas
threshold = 0.5  # Adjust the threshold as per your needs
y_pred = np.where(predictions >= threshold, 1, 0)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, predictions)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall:", recall)
print("F1 Score: ", f1)
print("ROC AUC:", roc_auc)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
classes = ['Tinto', 'Blanco']
disp = ConfusionMatrixDisplay(cm, display_labels=classes)
disp.plot()
plt.savefig("cm.png")