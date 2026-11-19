"""
Evaluación final independiente del modelo predictivo de riesgo OSS Guard.
Etapa 3 - Cierre Técnico: Evaluación sobre partición estricta por paquetes (Group Split).

Responsabilidad exclusiva:
- Cargar particiones regeneradas (train, validation, test) sin solapamiento de paquetes.
- Verificar integridad y ausencia de fuga de características.
- Entrenar el modelo seleccionado en validación (Regresión Logística) sobre datos autorizados (train + validation).
- Evaluar una sola vez sobre el conjunto de prueba independiente test.csv.
- Calcular y validar métricas obligatorias y matriz de confusión (TN, FP, FN, TP).
- Generar análisis detallado de falsos negativos.
- Exportar resultados y metadatos trazables con hashes SHA-256.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ============================================================
# CONFIGURACIÓN Y RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ARCHIVO_TRAIN = BASE_DIR / "data" / "processed" / "train.csv"
ARCHIVO_VALIDACION = BASE_DIR / "data" / "processed" / "validation.csv"
ARCHIVO_TEST = BASE_DIR / "data" / "processed" / "test.csv"

CARPETA_RESULTADOS = BASE_DIR / "data" / "results"
ARCHIVO_METRICAS = CARPETA_RESULTADOS / "evaluacion_final.csv"
ARCHIVO_METRICAS_E3 = CARPETA_RESULTADOS / "evaluacion_final_etapa3.csv"
ARCHIVO_MATRIZ = CARPETA_RESULTADOS / "matriz_confusion_final.csv"
ARCHIVO_MATRIZ_E3 = CARPETA_RESULTADOS / "matriz_confusion_etapa3.csv"
ARCHIVO_PREDICCIONES = CARPETA_RESULTADOS / "predicciones_test.csv"
ARCHIVO_PREDICCIONES_E3 = CARPETA_RESULTADOS / "predicciones_test_etapa3.csv"
ARCHIVO_FALSOS_NEGATIVOS = CARPETA_RESULTADOS / "falsos_negativos_test.csv"
ARCHIVO_FALSOS_NEGATIVOS_E3 = CARPETA_RESULTADOS / "falsos_negativos_test_etapa3.csv"
ARCHIVO_METADATOS = CARPETA_RESULTADOS / "metadata_evaluacion_final.json"
ARCHIVO_METADATOS_E3 = CARPETA_RESULTADOS / "metadata_evaluacion_final_etapa3.json"

RANDOM_STATE = 42

CARACTERISTICAS = [
    "edad_version_dias",
    "versiones_anteriores",
    "numero_dependencias",
    "numero_dependencias_opcionales",
    "numero_scripts",
    "tiene_repositorio",
    "tiene_licencia",
]

VARIABLE_OBJETIVO = "riesgo"

COLUMNAS_PROHIBIDAS_PREDICCION = [
    "riesgo",
    "numero_vulnerabilidades",
    "paquete",
    "version",
]


def calcular_hash_archivo(ruta: Path) -> str:
    """Calcula el hash SHA-256 de un archivo para verificar inalterabilidad."""
    hasher = hashlib.sha256()
    with open(ruta, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print()
    print("=" * 65)
    print("OSS GUARD - EVALUACIÓN FINAL INDEPENDIENTE (ETAPA 3)")
    print("EVALUACIÓN SOBRE PARTICIÓN ESTRATIFICADA POR PAQUETES")
    print("=" * 65)
    print(f"Fecha y hora: {datetime.now(timezone.utc).isoformat()}")
    print()

    # ------------------------------------------------------------
    # 1. VERIFICACIÓN DE EXISTENCIA Y CARGA DE PARTICIONES
    # ------------------------------------------------------------
    for ruta_archivo, nombre in [
        (ARCHIVO_TRAIN, "train"),
        (ARCHIVO_VALIDACION, "validation"),
        (ARCHIVO_TEST, "test"),
    ]:
        if not ruta_archivo.is_file():
            print(f"ERROR CRÍTICO: No se encontró el archivo {nombre}: {ruta_archivo}")
            sys.exit(1)

    hash_train_antes = calcular_hash_archivo(ARCHIVO_TRAIN)
    hash_val_antes = calcular_hash_archivo(ARCHIVO_VALIDACION)
    hash_test_antes = calcular_hash_archivo(ARCHIVO_TEST)

    train = pd.read_csv(ARCHIVO_TRAIN)
    validation = pd.read_csv(ARCHIVO_VALIDACION)
    test = pd.read_csv(ARCHIVO_TEST)

    print("Particiones cargadas:")
    print(f" - Entrenamiento : {len(train):>5} registros, {train['paquete'].nunique()} paquetes")
    print(f" - Validación    : {len(validation):>5} registros, {validation['paquete'].nunique()} paquetes")
    print(f" - Prueba (Test) : {len(test):>5} registros, {test['paquete'].nunique()} paquetes")
    print()

    # ------------------------------------------------------------
    # 2. VALIDACIÓN DE ETIQUETAS, NULOS Y DUPLICADOS
    # ------------------------------------------------------------
    particiones = {"train": train, "validation": validation, "test": test}
    conteos_clases = {}

    for nombre, df in particiones.items():
        if VARIABLE_OBJETIVO not in df.columns:
            print(f"ERROR CRÍTICO: La columna objetivo '{VARIABLE_OBJETIVO}' falta en {nombre}.")
            sys.exit(1)

        valores_invalidos = set(df[VARIABLE_OBJETIVO].dropna().unique()) - {0, 1}
        if valores_invalidos:
            print(f"ERROR CRÍTICO: Valores inválidos en '{VARIABLE_OBJETIVO}' ({nombre}): {valores_invalidos}")
            sys.exit(1)

        nulos_obj = df[VARIABLE_OBJETIVO].isnull().sum()
        if nulos_obj > 0:
            print(f"ERROR CRÍTICO: Se encontraron {nulos_obj} valores nulos en '{VARIABLE_OBJETIVO}' ({nombre}).")
            sys.exit(1)

        conteos = df[VARIABLE_OBJETIVO].value_counts().to_dict()
        conteos_clases[nombre] = {
            "clase_0_sin_riesgo": int(conteos.get(0, 0)),
            "clase_1_con_riesgo": int(conteos.get(1, 0)),
            "total": int(len(df)),
            "paquetes_unicos": int(df["paquete"].nunique()),
            "nulos_totales": int(df.isnull().sum().sum()),
            "duplicados_completos": int(df.duplicated().sum()),
            "duplicados_paquete_version": int(df.duplicated(subset=["paquete", "version"]).sum()),
        }

    print("Distribución de clases:")
    for nombre, c in conteos_clases.items():
        pct_0 = c["clase_0_sin_riesgo"] / c["total"] * 100
        pct_1 = c["clase_1_con_riesgo"] / c["total"] * 100
        print(f" - {nombre:<12}: Clase 0={c['clase_0_sin_riesgo']} ({pct_0:.2f}%), "
              f"Clase 1={c['clase_1_con_riesgo']} ({pct_1:.2f}%), "
              f"Paquetes={c['paquetes_unicos']}, Duplicados={c['duplicados_paquete_version']}")
    print()

    # ------------------------------------------------------------
    # 3. VERIFICACIÓN DE INDEPENDENCIA DE PAQUETES (CERO SOLAPAMIENTO)
    # ------------------------------------------------------------
    p_train = set(train["paquete"].dropna().unique())
    p_val = set(validation["paquete"].dropna().unique())
    p_test = set(test["paquete"].dropna().unique())

    compartidos_train_test = p_train & p_test
    compartidos_val_test = p_val & p_test
    compartidos_train_val = p_train & p_val

    print("Verificación de independencia de paquetes:")
    print(f" - Paquetes en Train y Test      : {len(compartidos_train_test)}")
    print(f" - Paquetes en Val y Test        : {len(compartidos_val_test)}")
    print(f" - Paquetes en Train y Val       : {len(compartidos_train_val)}")

    if len(compartidos_train_test) > 0 or len(compartidos_val_test) > 0 or len(compartidos_train_val) > 0:
        print("ERROR CRÍTICO: Solapamiento de paquetes detectado.")
        sys.exit(1)
    else:
        print(" [OK] Independencia total por paquetes confirmada. Solapamiento = 0.")
    print()

    # ------------------------------------------------------------
    # 4. PREVENCIÓN DE FUGA DE INFORMACIÓN EN CARACTERÍSTICAS
    # ------------------------------------------------------------
    print("Verificación de características predictoras:")
    for feat in CARACTERISTICAS:
        if feat in COLUMNAS_PROHIBIDAS_PREDICCION:
            print(f"BLOQUEO: La característica '{feat}' filtra información prohibida.")
            sys.exit(1)
        if feat not in train.columns or feat not in validation.columns or feat not in test.columns:
            print(f"BLOQUEO: La característica '{feat}' falta en los archivos procesados.")
            sys.exit(1)
        print(f" [OK] Característica validada: {feat}")
    print()

    # ------------------------------------------------------------
    # 5. PROTOCOLO DE ENTRENAMIENTO DEL MODELO FINAL
    # ------------------------------------------------------------
    # De acuerdo con la comparación de validación en la Etapa 3:
    # Regresión Logística obtuvo el mayor F1-score (0.7459) y alto Recall (83.50%).
    # Ajustamos la configuración seleccionada con train + validation (2.993 registros).
    train_final = pd.concat([train, validation], ignore_index=True)

    print("Preparación de datos para el modelo final:")
    print(f" - Registros de entrenamiento final (train + val): {len(train_final)} ({train_final['paquete'].nunique()} paquetes)")
    print(f" - Registros reservados exclusivamente para test : {len(test)} ({test['paquete'].nunique()} paquetes)")
    print()

    X_train_final = train_final[CARACTERISTICAS]
    y_train_final = train_final[VARIABLE_OBJETIVO]

    X_test = test[CARACTERISTICAS]
    y_test = test[VARIABLE_OBJETIVO]

    # Modelo seleccionado por validación
    modelo = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    hiperparametros = {"max_iter": 2000, "random_state": RANDOM_STATE}

    print("Configuración del modelo final (seleccionado en validación):")
    print(" - Algoritmo       : LogisticRegression")
    print(f" - Hiperparámetros : {hiperparametros}")
    print(f" - Semilla         : {RANDOM_STATE}")
    print()

    print("Ajustando modelo final en datos autorizados...")
    modelo.fit(X_train_final, y_train_final)
    print("Modelo ajustado correctamente.")
    print()

    # ------------------------------------------------------------
    # 6. EVALUACIÓN ÚNICA SOBRE EL CONJUNTO DE PRUEBA
    # ------------------------------------------------------------
    print("Ejecutando evaluación única sobre test.csv...")
    predicciones = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)

    clases = list(modelo.classes_)
    idx_clase_1 = clases.index(1) if 1 in clases else 1
    probabilidades_riesgo = probabilidades[:, idx_clase_1]

    # Métricas obligatorias
    acc = float(accuracy_score(y_test, predicciones))
    prec = float(precision_score(y_test, predicciones, zero_division=0))
    rec = float(recall_score(y_test, predicciones, zero_division=0))
    f1 = float(f1_score(y_test, predicciones, zero_division=0))

    matriz = confusion_matrix(y_test, predicciones, labels=[0, 1])
    tn = int(matriz[0, 0])
    fp = int(matriz[0, 1])
    fn = int(matriz[1, 0])
    tp = int(matriz[1, 1])
    total_evaluados = len(y_test)

    # ------------------------------------------------------------
    # 7. VALIDACIÓN ARITMÉTICA Y DE INTEGRIDAD
    # ------------------------------------------------------------
    suma_matriz = tn + fp + fn + tp
    if suma_matriz != total_evaluados:
        print(f"ERROR: Suma de matriz ({suma_matriz}) != total evaluado ({total_evaluados})")
        sys.exit(1)

    for nombre_m, val_m in [("Accuracy", acc), ("Precision", prec), ("Recall", rec), ("F1-score", f1)]:
        if not (0.0 <= val_m <= 1.0):
            print(f"ERROR: Métrica {nombre_m} ({val_m}) fuera del rango [0, 1]")
            sys.exit(1)

    # Validar que los CSVs de entrada permanezcan inalterados
    if calcular_hash_archivo(ARCHIVO_TRAIN) != hash_train_antes:
        print("ERROR CRÍTICO: train.csv fue alterado durante la ejecución.")
        sys.exit(1)
    if calcular_hash_archivo(ARCHIVO_VALIDACION) != hash_val_antes:
        print("ERROR CRÍTICO: validation.csv fue alterado durante la ejecución.")
        sys.exit(1)
    if calcular_hash_archivo(ARCHIVO_TEST) != hash_test_antes:
        print("ERROR CRÍTICO: test.csv fue alterado durante la ejecución.")
        sys.exit(1)

    print("Verificaciones aritméticas y de integridad superadas exitosamente.")
    print()

    # ------------------------------------------------------------
    # 8. PRESENTACIÓN DE RESULTADOS
    # ------------------------------------------------------------
    print("=" * 65)
    print("MÉTRICAS OBLIGATORIAS EN EL CONJUNTO DE PRUEBA (TEST)")
    print("=" * 65)
    print(f"Modelo evaluado : LogisticRegression (Seleccionado en validación)")
    print(f"Accuracy        : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision       : {prec:.4f} ({prec * 100:.2f}%)")
    print(f"Recall          : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"F1-score        : {f1:.4f} ({f1 * 100:.2f}%)")
    print()
    print("Matriz de confusión:")
    print(f"                 Predicción 0 (Sin riesgo)   Predicción 1 (Con riesgo)")
    print(f"Real 0 (Sin)     TN = {tn:<24} FP = {fp}")
    print(f"Real 1 (Con)     FN = {fn:<24} TP = {tp}")
    print(f"Total evaluados: {suma_matriz}")
    print()

    # También calculamos como referencia comparativa complementaria Random Forest sobre el mismo test
    rf_comparativo = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf_comparativo.fit(X_train_final, y_train_final)
    preds_rf = rf_comparativo.predict(X_test)
    acc_rf = float(accuracy_score(y_test, preds_rf))
    prec_rf = float(precision_score(y_test, preds_rf, zero_division=0))
    rec_rf = float(recall_score(y_test, preds_rf, zero_division=0))
    f1_rf = float(f1_score(y_test, preds_rf, zero_division=0))
    cm_rf = confusion_matrix(y_test, preds_rf, labels=[0, 1])

    print("Comparativa de referencia con Random Forest sobre el mismo Test:")
    print(f" - Random Forest : Accuracy={acc_rf:.4f}, Precision={prec_rf:.4f}, Recall={rec_rf:.4f}, F1={f1_rf:.4f}")
    print(f"   Matriz RF     : TN={cm_rf[0,0]}, FP={cm_rf[0,1]}, FN={cm_rf[1,0]}, TP={cm_rf[1,1]}")
    print()

    # ------------------------------------------------------------
    # 9. ANÁLISIS ESPECÍFICO DE FALSOS NEGATIVOS
    # ------------------------------------------------------------
    positivos_reales = fn + tp
    tasa_fn = (fn / positivos_reales * 100) if positivos_reales > 0 else 0.0

    print("=" * 65)
    print("ANÁLISIS DE FALSOS NEGATIVOS (MODELO SELECCIONADO)")
    print("=" * 65)
    print(f"Total de versiones riesgosas reales en test : {positivos_reales}")
    print(f"Versiones riesgosas clasificadas como NO riesgosas (FN): {fn}")
    print(f"Proporción de falsos negativos respecto a positivos   : {tasa_fn:.2f}% ({fn}/{positivos_reales})")
    print()

    df_predicciones = test.copy()
    df_predicciones["prediccion_riesgo"] = predicciones
    df_predicciones["probabilidad_riesgo"] = np.round(probabilidades_riesgo, 4)
    df_predicciones["acierto"] = (df_predicciones[VARIABLE_OBJETIVO] == df_predicciones["prediccion_riesgo"])

    falsos_negativos_df = df_predicciones[
        (df_predicciones[VARIABLE_OBJETIVO] == 1) & (df_predicciones["prediccion_riesgo"] == 0)
    ].copy()

    print(f"Se identificaron {len(falsos_negativos_df)} casos de falsos negativos en test.")
    if len(falsos_negativos_df) > 0:
        cols_fn = ["paquete", "version", "edad_version_dias", "versiones_anteriores",
                   "numero_dependencias", "probabilidad_riesgo"]
        print("\nDistribución de FN por paquete:")
        print(falsos_negativos_df["paquete"].value_counts().to_string())
        print("\nMuestra de falsos negativos:")
        print(falsos_negativos_df[cols_fn].head(15).to_string(index=False))
        print()

    # ------------------------------------------------------------
    # 10. GUARDAR RESULTADOS Y METADATOS
    # ------------------------------------------------------------
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    # 10.1 Métricas finales
    df_metricas = pd.DataFrame([
        {
            "modelo": "Regresion Logistica",
            "conjunto": "test",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        },
        {
            "modelo": "Random Forest",
            "conjunto": "test_referencia",
            "accuracy": acc_rf,
            "precision": prec_rf,
            "recall": rec_rf,
            "f1_score": f1_rf,
        },
    ])
    df_metricas.to_csv(ARCHIVO_METRICAS, index=False, encoding="utf-8")
    df_metricas.to_csv(ARCHIVO_METRICAS_E3, index=False, encoding="utf-8")
    print(f" [GUARDADO] Métricas finales               : {ARCHIVO_METRICAS}")

    # 10.2 Matriz de confusión
    df_matriz = pd.DataFrame([
        {
            "modelo": "Regresion Logistica",
            "verdaderos_negativos": tn,
            "falsos_positivos": fp,
            "falsos_negativos": fn,
            "verdaderos_positivos": tp,
            "total_evaluados": total_evaluados,
        },
        {
            "modelo": "Random Forest",
            "verdaderos_negativos": int(cm_rf[0, 0]),
            "falsos_positivos": int(cm_rf[0, 1]),
            "falsos_negativos": int(cm_rf[1, 0]),
            "verdaderos_positivos": int(cm_rf[1, 1]),
            "total_evaluados": total_evaluados,
        },
    ])
    df_matriz.to_csv(ARCHIVO_MATRIZ, index=False, encoding="utf-8")
    df_matriz.to_csv(ARCHIVO_MATRIZ_E3, index=False, encoding="utf-8")
    print(f" [GUARDADO] Matriz de confusión            : {ARCHIVO_MATRIZ}")

    # 10.3 Predicciones completas del test
    cols_pred_salida = [
        "paquete",
        "version",
        "edad_version_dias",
        "versiones_anteriores",
        "numero_dependencias",
        "numero_dependencias_opcionales",
        "numero_scripts",
        "tiene_repositorio",
        "tiene_licencia",
        "riesgo_real",
        "prediccion_riesgo",
        "probabilidad_riesgo",
        "acierto",
    ]
    df_pred_exportar = df_predicciones.rename(columns={VARIABLE_OBJETIVO: "riesgo_real"}).copy()
    df_pred_exportar[cols_pred_salida].to_csv(ARCHIVO_PREDICCIONES, index=False, encoding="utf-8")
    df_pred_exportar[cols_pred_salida].to_csv(ARCHIVO_PREDICCIONES_E3, index=False, encoding="utf-8")
    print(f" [GUARDADO] Predicciones individuales test : {ARCHIVO_PREDICCIONES}")

    # 10.4 Detalle de falsos negativos para auditoría
    falsos_negativos_df.rename(columns={VARIABLE_OBJETIVO: "riesgo_real"})[cols_pred_salida].to_csv(
        ARCHIVO_FALSOS_NEGATIVOS, index=False, encoding="utf-8"
    )
    falsos_negativos_df.rename(columns={VARIABLE_OBJETIVO: "riesgo_real"})[cols_pred_salida].to_csv(
        ARCHIVO_FALSOS_NEGATIVOS_E3, index=False, encoding="utf-8"
    )
    print(f" [GUARDADO] Auditoría de falsos negativos  : {ARCHIVO_FALSOS_NEGATIVOS}")

    # 10.5 Metadatos de evaluación y trazabilidad
    metadata = {
        "fecha_ejecucion": datetime.now(timezone.utc).isoformat(),
        "proyecto": "OSS Guard",
        "etapa": "Etapa 3 - Cierre Tecnico",
        "estrategia_particion": "Group Split por paquete (0 solapamiento entre particiones)",
        "particiones": {
            "train": {
                "ruta": str(ARCHIVO_TRAIN.relative_to(BASE_DIR)),
                "filas": len(train),
                "paquetes_unicos": int(train["paquete"].nunique()),
                "sha256": hash_train_antes,
                "distribucion_clases": conteos_clases["train"],
            },
            "validation": {
                "ruta": str(ARCHIVO_VALIDACION.relative_to(BASE_DIR)),
                "filas": len(validation),
                "paquetes_unicos": int(validation["paquete"].nunique()),
                "sha256": hash_val_antes,
                "distribucion_clases": conteos_clases["validation"],
            },
            "test": {
                "ruta": str(ARCHIVO_TEST.relative_to(BASE_DIR)),
                "filas": len(test),
                "paquetes_unicos": int(test["paquete"].nunique()),
                "sha256": hash_test_antes,
                "distribucion_clases": conteos_clases["test"],
            },
            "train_final_combinado": {
                "filas": len(train_final),
                "paquetes_unicos": int(train_final["paquete"].nunique()),
                "descripcion": "train.csv + validation.csv segun protocolo para ajustar el modelo seleccionado en validacion",
            },
        },
        "modelo_seleccionado": {
            "algoritmo": "LogisticRegression",
            "regla_seleccion": "Mayor F1-score en validation (0.7459) con alto Recall (83.50%)",
            "semilla": RANDOM_STATE,
            "hiperparametros": hiperparametros,
            "caracteristicas_utilizadas": CARACTERISTICAS,
            "variable_objetivo": VARIABLE_OBJETIVO,
            "umbral_decision": 0.5,
        },
        "resultados_test_modelo_seleccionado": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "matriz_confusion": {
                "verdaderos_negativos": tn,
                "falsos_positivos": fp,
                "falsos_negativos": fn,
                "verdaderos_positivos": tp,
                "total": total_evaluados,
            },
            "falsos_negativos": {
                "cantidad": fn,
                "positivos_reales": positivos_reales,
                "proporcion_porcentual": tasa_fn,
            },
        },
        "referencia_random_forest_test": {
            "accuracy": acc_rf,
            "precision": prec_rf,
            "recall": rec_rf,
            "f1_score": f1_rf,
            "matriz_confusion": {
                "verdaderos_negativos": int(cm_rf[0, 0]),
                "falsos_positivos": int(cm_rf[0, 1]),
                "falsos_negativos": int(cm_rf[1, 0]),
                "verdaderos_positivos": int(cm_rf[1, 1]),
            },
        },
        "verificaciones_metodologicas": {
            "ausencia_fuga_caracteristicas": True,
            "columnas_excluidas_predictoras": COLUMNAS_PROHIBIDAS_PREDICCION,
            "declaracion_independencia_test": "verificado_en_etapa3",
            "detalle_independencia": (
                "En esta Etapa 3 las particiones se regeneraron estrictamente por grupos de paquetes a partir de dataset_ml.csv. "
                "Los 11 paquetes evaluados en test.csv no tienen ninguna version en train ni en validation (solapamiento = 0). "
                "El modelo fue seleccionado exclusivamente mediante la comparacion en validation.csv sin consultar test.csv."
            ),
            "solapamiento_paquetes": 0,
        },
    }

    with open(ARCHIVO_METADATOS, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(ARCHIVO_METADATOS_E3, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f" [GUARDADO] Metadatos de auditoría        : {ARCHIVO_METADATOS}")

    print()
    print("=" * 65)
    print("EVALUACIÓN FINAL COMPLETADA EXITOSAMENTE (ETAPA 3)")
    print("=" * 65)


if __name__ == "__main__":
    main()
