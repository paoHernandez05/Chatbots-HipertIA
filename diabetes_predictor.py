"""
predict_diabetes.py
-------------------
Módulo de predicción de riesgo de diabetes.
Carga los artefactos (.pkl) una vez al importar y expone la función predecir_diabetes().

Artefactos requeridos en la misma carpeta:
    - diabetes_rf_model.pkl
    - diabetes_shap_explainer.pkl
    - diabetes_feature_names.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Carga de artefactos (una sola vez al importar)
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_model         = joblib.load(os.path.join(_BASE_DIR, "diabetes_rf_model.pkl"))
_explainer     = joblib.load(os.path.join(_BASE_DIR, "diabetes_shap_explainer.pkl"))
_feature_names = joblib.load(os.path.join(_BASE_DIR, "diabetes_feature_names.pkl"))

# ─────────────────────────────────────────────
# Constantes y mapas
# ─────────────────────────────────────────────
_LABELS       = {0: "Bajo Riesgo", 1: "Prediabetes", 2: "Alto Riesgo"}
_NIVEL_RIESGO = {0: "Bajo",        1: "Moderado",    2: "Alto"}

_FEATURE_LABELS_SHORT = {
    "age":                        "Edad",
    "gender":                     "Sexo",
    "bmi":                        "IMC",
    "blood_pressure":             "Presión arterial",
    "insulin_level":              "Insulina",
    "cholesterol_level":          "Colesterol",
    "triglycerides_level":        "Triglicéridos",
    "physical_activity_level":    "Actividad física",
    "daily_calorie_intake":       "Calorías diarias",
    "sugar_intake_grams_per_day": "Consumo de azúcar",
    "sleep_hours":                "Horas de sueño",
    "stress_level":               "Estrés",
    "family_history_diabetes":    "Antecedentes familiares",
    "waist_circumference_cm":     "Circunferencia abdominal",
}

_CORRECTION_THRESHOLD = {
    "physical_activity_level":    0.15,
    "stress_level":               0.10,
    "sleep_hours":                0.10,
    "sugar_intake_grams_per_day": 0.08,
    "waist_circumference_cm":     0.08,
    "bmi":                        0.06,
    "triglycerides_level":        0.06,
    "daily_calorie_intake":       0.05,
    "blood_pressure":             0.05,
    "cholesterol_level":          0.05,
    "insulin_level":              0.05,
    "age":                        0.04,
    "family_history_diabetes":    0.04,
    "gender":                     0.00,
}

_SAFE_RANGE   = {"sleep_hours": (6.0, 9.0)}
_STRESS_ZONES = {"low": (1, 4), "high": (7, 10)}

_RECOMENDACIONES_MAP = {
    "bmi":                        "Reducir peso hacia un IMC saludable (18.5–24.9)",
    "blood_pressure":             "Control periódico de presión arterial",
    "sugar_intake_grams_per_day": "Reducir consumo de azúcar y alimentos ultraprocesados",
    "physical_activity_level":    "Incrementar actividad física de forma regular",
    "waist_circumference_cm":     "Reducir circunferencia abdominal con dieta y ejercicio",
    "stress_level":               "Implementar técnicas de manejo del estrés",
    "sleep_hours":                "Ajustar hábitos de sueño al rango saludable (6–9 horas)",
    "triglycerides_level":        "Reducir triglicéridos con dieta baja en carbohidratos refinados",
    "cholesterol_level":          "Controlar niveles de colesterol con dieta y ejercicio",
    "daily_calorie_intake":       "Moderar la ingesta calórica diaria",
    "insulin_level":              "Seguimiento médico de niveles de insulina",
    "age":                        "Control médico preventivo periódico por edad",
    "family_history_diabetes":    "Seguimiento médico por antecedentes familiares de diabetes",
}


# ─────────────────────────────────────────────
# Corrección clínica de SHAP
# ─────────────────────────────────────────────
def _clinical_effect(variable: str, valor: float, shap_impacto: float):
    """Corrige la dirección del SHAP cuando contradice la evidencia clínica y la señal es débil."""
    umbral   = _CORRECTION_THRESHOLD.get(variable, 0.0)
    abs_imp  = abs(shap_impacto)
    es_ruido = abs_imp < umbral

    if variable == "physical_activity_level":
        if int(valor) == 2:
            correcto = shap_impacto <= 0;  umbral_local = umbral
        elif int(valor) == 1:
            correcto = shap_impacto <= 0;  umbral_local = umbral * 0.5
        else:
            correcto = shap_impacto >= 0;  umbral_local = umbral
        if not correcto and abs_imp < umbral_local:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "sleep_hours":
        lo, hi   = _SAFE_RANGE["sleep_hours"]
        en_rango = lo <= valor <= hi
        correcto = shap_impacto <= 0 if en_rango else shap_impacto >= 0
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "stress_level":
        if valor <= _STRESS_ZONES["low"][1]:
            correcto = shap_impacto <= 0
        elif valor >= _STRESS_ZONES["high"][0]:
            correcto = shap_impacto >= 0
        else:
            correcto = True
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    _higher_risk = {
        "sugar_intake_grams_per_day", "triglycerides_level", "waist_circumference_cm",
        "bmi", "daily_calorie_intake", "blood_pressure", "cholesterol_level",
        "insulin_level", "age", "family_history_diabetes",
    }
    if variable in _higher_risk:
        correcto = shap_impacto >= 0
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    return shap_impacto, False


def _severidad(contrib_pct_abs: float) -> str:
    if contrib_pct_abs >= 20: return "Alto"
    if contrib_pct_abs >= 8:  return "Moderado"
    return "Bajo"


# ─────────────────────────────────────────────
# Función principal de predicción
# ─────────────────────────────────────────────
def predecir_diabetes(datos: dict) -> dict:
    """
    Predice el riesgo de diabetes de un paciente.

    Parámetros (claves del dict):
        age                        float  Edad en años
        gender                     int    0=Femenino  1=Masculino
        bmi                        float  IMC (kg/m²)
        blood_pressure             float  Presión sistólica (mmHg) ← measurement.systolic_bp en BD
        insulin_level              float  Insulina (µU/mL)
        cholesterol_level          float  Colesterol total (mg/dL)
        triglycerides_level        float  Triglicéridos (mg/dL)
        physical_activity_level    int    0=Sin actividad  1=Algo semanal  2=Constante
        daily_calorie_intake       float  Calorías diarias (kcal)
        sugar_intake_grams_per_day float  Azúcar (g/día)
        sleep_hours                float  Horas de sueño
        stress_level               int    Estrés (1–10)
        family_history_diabetes    int    0=No  1=Sí
        waist_circumference_cm     float  Circunferencia de cintura (cm)

    Retorna dict con: puntuacion, clasificacion, nivel_riesgo,
                      probabilidades, factores_principales, recomendaciones
    """
    paciente         = pd.DataFrame([datos])[_feature_names]
    valores_paciente = paciente.values[0]

    pred_class = int(_model.predict(paciente)[0])
    pred_proba = _model.predict_proba(paciente)[0]

    # SHAP — siempre referenciado a clase 2 (Alto Riesgo)
    shap_values = _explainer.shap_values(paciente)
    if isinstance(shap_values, list):
        shap_raw = shap_values[2][0]
    else:
        shap_raw = shap_values[0, :, 2]

    shap_corr, fue_corr = [], []
    for i, var in enumerate(_feature_names):
        sc, fc = _clinical_effect(var, valores_paciente[i], shap_raw[i])
        shap_corr.append(sc)
        fue_corr.append(fc)

    shap_corr   = np.array(shap_corr)
    total_abs   = np.sum(np.abs(shap_corr))
    contrib_pct = (shap_corr / total_abs * 100) if total_abs > 0 else shap_corr

    impacto_df = pd.DataFrame({
        "Variable":    _feature_names,
        "Valor":       valores_paciente,
        "Impacto":     shap_corr,
        "Contrib_pct": contrib_pct,
        "Corregido":   fue_corr,
    }).sort_values("Impacto", key=abs, ascending=False)

    puntuacion = min(100, round(float(pred_proba[1]) * 50 + float(pred_proba[2]) * 100))

    # Top 4 factores (excluye sexo que no tiene dirección clínica clara)
    top_factores = impacto_df[impacto_df["Variable"] != "gender"].head(4)
    factores_ui  = [
        {
            "nombre":    _FEATURE_LABELS_SHORT.get(r["Variable"], r["Variable"]),
            "valor":     round(float(r["Valor"]), 1),
            "efecto":    "riesgo" if r["Impacto"] > 0 else "protector",
            "severidad": _severidad(abs(r["Contrib_pct"])),
            "peso_pct":  round(abs(float(r["Contrib_pct"])), 1),
        }
        for _, r in top_factores.iterrows()
    ]

    recomendaciones = []
    for _, row in impacto_df.iterrows():
        if row["Impacto"] > 0 and row["Variable"] in _RECOMENDACIONES_MAP:
            recomendaciones.append(_RECOMENDACIONES_MAP[row["Variable"]])
        if len(recomendaciones) >= 4:
            break
    if not recomendaciones:
        recomendaciones.append("Mantener los hábitos actuales de vida saludable")
    if len(recomendaciones) < 2:
        recomendaciones.append("Control médico anual preventivo")

    return {
        "puntuacion":     int(puntuacion),
        "clasificacion":  _LABELS[pred_class],
        "nivel_riesgo":   _NIVEL_RIESGO[pred_class],
        "probabilidades": {
            "bajo_riesgo": round(float(pred_proba[0]) * 100, 1),
            "prediabetes":  round(float(pred_proba[1]) * 100, 1),
            "alto_riesgo":  round(float(pred_proba[2]) * 100, 1),
        },
        "factores_principales": factores_ui,
        "recomendaciones":      recomendaciones,
    }