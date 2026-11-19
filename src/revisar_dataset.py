import pandas as pd

archivo = "data/raw/dataset_npm.csv"

df = pd.read_csv(archivo)

print("\n========== INFORMACIÓN DEL DATASET ==========\n")

print("Número de filas:", df.shape[0])
print("Número de columnas:", df.shape[1])

print("\nColumnas:")
print(df.columns.tolist())

print("\nPrimeros registros:")
print(df.head())

print("\nValores faltantes:")
print(df.isnull().sum())

print("\nTipos de datos:")
print(df.dtypes)

print("\n============================================")