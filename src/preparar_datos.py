import os
from pathlib import Path
import numpy as np
import pandas as pd

# ==========================================
# ARCHIVOS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_ENTRADA = BASE_DIR / "data" / "processed" / "dataset_ml.csv"
ARCHIVO_TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
ARCHIVO_VALIDACION = BASE_DIR / "data" / "processed" / "validation.csv"
ARCHIVO_TEST = BASE_DIR / "data" / "processed" / "test.csv"

# ==========================================
# CONFIGURACIÓN
# ==========================================

RANDOM_STATE = 42
os.makedirs(BASE_DIR / "data" / "processed", exist_ok=True)

print()
print("===================================")
print("PREPARANDO DATOS PARA MACHINE LEARNING")
print("SEPARACIÓN POR GRUPOS DE PAQUETES (ETAPA 3)")
print("===================================")
print()

# ==========================================
# CARGAR DATASET
# ==========================================

dataset = pd.read_csv(ARCHIVO_ENTRADA)
registros_origen = len(dataset)
print(f"Registros totales de origen: {registros_origen}")
print()

# ==========================================
# VERIFICAR COLUMNAS
# ==========================================

columnas_esperadas = [
    "paquete",
    "version",
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
    "riesgo",
]

columnas_faltantes = [col for col in columnas_esperadas if col not in dataset.columns]
if columnas_faltantes:
    print(f"ERROR CRÍTICO: faltan las siguientes columnas: {columnas_faltantes}")
    raise SystemExit(1)

# ==========================================
# LIMPIEZA DE DATOS Y COMPROBACIONES
# ==========================================

duplicados_iniciales = dataset.duplicated(subset=["paquete", "version"]).sum()
if duplicados_iniciales > 0:
    print(f"Aviso: Se eliminaron {duplicados_iniciales} registros duplicados por (paquete, version).")
    dataset = dataset.drop_duplicates(subset=["paquete", "version"])

nulos_iniciales = dataset[columnas_esperadas].isnull().sum().sum()
if nulos_iniciales > 0:
    print(f"Aviso: Se eliminaron filas con nulos en columnas esperadas.")
    dataset = dataset.dropna(subset=columnas_esperadas)

dataset = dataset[dataset["riesgo"].isin([0, 1])].copy()
dataset["riesgo"] = dataset["riesgo"].astype(int)

print(f"Registros válidos tras validación: {len(dataset)}")
print("Distribución total de clases:")
print(dataset["riesgo"].value_counts().sort_index())
print()

# ==========================================
# SEPARACIÓN POR PAQUETE (GROUP SPLIT ESTRATIFICADO)
# ==========================================
#
# Requisito obligatorio: Todas las versiones de un mismo paquete
# deben permanecer en una única partición. Ninguna versión de un paquete
# puede aparecer en más de una partición.
#
# Target de partición: ~60% train, ~20% validación, ~20% prueba.
# ==========================================

stats = (
    dataset.groupby("paquete")
    .agg(
        total=("riesgo", "count"),
        n_0=("riesgo", lambda x: (x == 0).sum()),
        n_1=("riesgo", lambda x: (x == 1).sum()),
    )
    .reset_index()
)

total_rows = len(dataset)
total_1 = (dataset["riesgo"] == 1).sum()
target_class_prop = total_1 / total_rows

target_train = 0.60
target_val = 0.20
target_test = 0.20

best_loss = float("inf")
best_assignment = None

rng = np.random.RandomState(RANDOM_STATE)

# Búsqueda determinista de asignación por grupos
for _ in range(50000):
    perm = rng.permutation(len(stats))
    t_rows, v_rows, te_rows = 0, 0, 0
    t_0, v_0, te_0 = 0, 0, 0
    t_1, v_1, te_1 = 0, 0, 0
    assignment = {}

    for idx in perm:
        row = stats.iloc[idx]
        p_name = row["paquete"]
        p_tot = row["total"]
        p0 = row["n_0"]
        p1 = row["n_1"]

        deficits = [
            (target_train * total_rows - (t_rows + p_tot)) / (target_train * total_rows),
            (target_val * total_rows - (v_rows + p_tot)) / (target_val * total_rows),
            (target_test * total_rows - (te_rows + p_tot)) / (target_test * total_rows),
        ]
        chosen = int(np.argmax(deficits))

        if chosen == 0:
            t_rows += p_tot
            t_0 += p0
            t_1 += p1
            assignment[p_name] = "train"
        elif chosen == 1:
            v_rows += p_tot
            v_0 += p0
            v_1 += p1
            assignment[p_name] = "validation"
        else:
            te_rows += p_tot
            te_0 += p0
            te_1 += p1
            assignment[p_name] = "test"

    if min(t_0, t_1, v_0, v_1, te_0, te_1) < 50:
        continue

    prop_t = t_rows / total_rows
    prop_v = v_rows / total_rows
    prop_te = te_rows / total_rows

    prop_t_1 = t_1 / t_rows if t_rows > 0 else 0
    prop_v_1 = v_1 / v_rows if v_rows > 0 else 0
    prop_te_1 = te_1 / te_rows if te_rows > 0 else 0

    loss_size = (prop_t - target_train) ** 2 + (prop_v - target_val) ** 2 + (prop_te - target_test) ** 2
    loss_class = (prop_t_1 - target_class_prop) ** 2 + (prop_v_1 - target_class_prop) ** 2 + (prop_te_1 - target_class_prop) ** 2
    total_loss = loss_size * 2.0 + loss_class

    if total_loss < best_loss:
        best_loss = total_loss
        best_assignment = assignment.copy()

dataset["particion"] = dataset["paquete"].map(best_assignment)

train = dataset[dataset["particion"] == "train"].drop(columns=["particion"]).copy()
validation = dataset[dataset["particion"] == "validation"].drop(columns=["particion"]).copy()
test = dataset[dataset["particion"] == "test"].drop(columns=["particion"]).copy()

# ==========================================
# VALIDACIONES OBLIGATORIAS
# ==========================================

pkgs_train = set(train["paquete"])
pkgs_val = set(validation["paquete"])
pkgs_test = set(test["paquete"])

print("Validación de separación estricta por paquetes:")
print(f" - Paquetes en Train      : {len(pkgs_train)}")
print(f" - Paquetes en Validation : {len(pkgs_val)}")
print(f" - Paquetes en Test       : {len(pkgs_test)}")
print(f" - Solapamiento Train-Val : {len(pkgs_train & pkgs_val)}")
print(f" - Solapamiento Train-Test: {len(pkgs_train & pkgs_test)}")
print(f" - Solapamiento Val-Test  : {len(pkgs_val & pkgs_test)}")

if (len(pkgs_train & pkgs_val) > 0) or (len(pkgs_train & pkgs_test) > 0) or (len(pkgs_val & pkgs_test) > 0):
    print("ERROR CRÍTICO: Se detectaron paquetes compartidos entre particiones.")
    raise SystemExit(1)

for nombre_df, df_part in [("train", train), ("validation", validation), ("test", test)]:
    if df_part.isnull().sum().sum() > 0:
        print(f"ERROR: Valores nulos detectados en {nombre_df}")
        raise SystemExit(1)
    if df_part.duplicated(subset=["paquete", "version"]).sum() > 0:
        print(f"ERROR: Duplicados detectados en {nombre_df}")
        raise SystemExit(1)
    c0 = (df_part["riesgo"] == 0).sum()
    c1 = (df_part["riesgo"] == 1).sum()
    if c0 == 0 or c1 == 0:
        print(f"ERROR: La partición {nombre_df} no contiene ambas clases.")
        raise SystemExit(1)

print("\nTodas las validaciones obligatorias fueron superadas exitosamente.")
print()

# ==========================================
# GUARDAR DATASETS
# ==========================================

train.to_csv(ARCHIVO_TRAIN, index=False, encoding="utf-8")
validation.to_csv(ARCHIVO_VALIDACION, index=False, encoding="utf-8")
test.to_csv(ARCHIVO_TEST, index=False, encoding="utf-8")

# ==========================================
# MOSTRAR RESUMEN FINAL
# ==========================================

total_p = len(dataset)
print("===================================")
print("SEPARACIÓN DE DATOS COMPLETADA (ETAPA 3)")
print("===================================")
print(f"Entrenamiento : {len(train):>5} registros ({len(train)/total_p*100:.2f}%) - Clase 0: {(train['riesgo']==0).sum()}, Clase 1: {(train['riesgo']==1).sum()}")
print(f"Validación    : {len(validation):>5} registros ({len(validation)/total_p*100:.2f}%) - Clase 0: {(validation['riesgo']==0).sum()}, Clase 1: {(validation['riesgo']==1).sum()}")
print(f"Prueba final  : {len(test):>5} registros ({len(test)/total_p*100:.2f}%) - Clase 0: {(test['riesgo']==0).sum()}, Clase 1: {(test['riesgo']==1).sum()}")
print()
print("Archivos generados:")
print(" -", ARCHIVO_TRAIN)
print(" -", ARCHIVO_VALIDACION)
print(" -", ARCHIVO_TEST)
print()
print("===================================")
print("PROCESO TERMINADO")
print("===================================")