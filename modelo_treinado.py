import pandas as pd
import rasterio
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ------------------------------
# 1) Carregar amostra de treino
# ------------------------------
amostra_csv = r"c:\Users\nadel_tumi658\Documents\Eng. Ambiental 2025.1\TCC I\codes test 1\planilha\amostras_indices.csv"
df = pd.read_csv(amostra_csv, sep=';')

print("Colunas do CSV:", df.columns)

# Separar atributos e classe
X = df.drop(columns=['fid', 'class'])
y = df['class']

# ------------------------------
# 2) Treinar Random Forest
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avaliar
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# ------------------------------
# 3) Classificar imagem Landsat
# ------------------------------
landsat_path = r"C:\Users\nadel_tumi658\Documents\Eng. Ambiental 2025.2\TCC_2\validações\validaçao_30_07_24\Landsat8_2024-07-30.tif"
output_path = r"C:\Users\nadel_tumi658\Documents\Eng. Ambiental 2025.2\TCC_2\validações\validação_30_04_7\classificacao_rf_4.tif"

with rasterio.open(landsat_path) as src:
    bands = src.read()  # shape: (n_bandas, altura, largura)
    profile = src.profile
    nodata = src.nodata

print("Bands shape antes do ajuste:", bands.shape)

# Garantir apenas 7 bandas puras
if bands.shape[0] > 7:
    bands = bands[:7, :, :]
print("Bands shape após ajuste:", bands.shape)

# Calcular índices
ndvi = (bands[4] - bands[3]) / (bands[4] + bands[3])
ndmi = (bands[4] - bands[5]) / (bands[4] + bands[5])
nbr  = (bands[4] - bands[6]) / (bands[4] + bands[6])

# Substituir valores inválidos (NaN/Inf) por 0
ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
ndmi = np.nan_to_num(ndmi, nan=0.0, posinf=0.0, neginf=0.0)
nbr  = np.nan_to_num(nbr, nan=0.0, posinf=0.0, neginf=0.0)

# Stack bandas + índices
stack = np.stack([bands[0], bands[1], bands[2], bands[3], bands[4], bands[5], bands[6],
                  ndvi, ndmi, nbr])  # shape (10, altura, largura)

n_bands, height, width = stack.shape
X_pred = stack.reshape(n_bands, -1).T

# Criar máscara para pixels válidos (somatório != 0)
mask_valido = np.any(stack != 0, axis=0).reshape(-1)

# Predição apenas para pixels válidos
y_pred_raster = np.zeros(height * width, dtype=np.uint8)
y_pred_raster[mask_valido] = clf.predict(X_pred[mask_valido])

y_pred_raster = y_pred_raster.reshape(height, width)

# ------------------------------
# 4) Visualizar antes de salvar
# ------------------------------
plt.figure(figsize=(8, 8))
plt.imshow(y_pred_raster, cmap='tab20')
plt.colorbar(label='Classe')
plt.title('Classificação Random Forest')
plt.axis('off')
plt.show()

# ------------------------------
# 5) Salvar raster classificado
# ------------------------------
profile.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(y_pred_raster, 1)

print("Classificação concluída e salva em:", output_path)
