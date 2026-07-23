"""
predict_hypertension.py
-----------------------
Módulo de predicción de riesgo de hipertensión arterial.
Carga los artefactos (.pkl) una vez al importar y expone la función predecir_hipertension().

Artefactos requeridos en la misma carpeta:
    - hypertension_rf_model.pkl
    - hypertension_shap_explainer.pkl
    - hypertension_feature_names.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Carga de artefactos (una sola vez al importar)
# ─────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_model         = joblib.load(os.path.join(_BASE_DIR, "hypertension_rf_model.pkl"))
_explainer     = joblib.load(os.path.join(_BASE_DIR, "hypertension_shap_explainer.pkl"))
_feature_names = joblib.load(os.path.join(_BASE_DIR, "hypertension_feature_names.pkl"))

# ─────────────────────────────────────────────
# Constantes y mapas
# ─────────────────────────────────────────────
_LABELS       = {0: "Sin Hipertensión", 1: "Prehipertensión", 2: "Con Hipertensión"}
_NIVEL_RIESGO = {0: "Bajo",             1: "Moderado",         2: "Alto"}

# Suavizado bayesiano simétrico aplicado a las probabilidades del bosque.
# Evita presentar 0 % o 100 % como certezas clínicas cuando los árboles votan
# de forma prácticamente unánime. No cambia la clase predicha ni el contrato.
_PROBABILITY_PRIOR_STRENGTH = 2.0

_FEATURE_LABELS_SHORT = {
    "Age":                     "Edad",
    "BMI":                     "IMC",
    "Cholesterol":              "Colesterol",
    "Smoking_Status":           "Tabaquismo",
    "Alcohol_Intake":           "Consumo de alcohol",
    "Physical_Activity_Level":  "Actividad física",
    "Family_History":           "Antecedentes familiares",
    "Diabetes":                 "Diabetes",
    "Stress_Level":             "Estrés",
    "Salt_Intake":              "Ingesta de sal",
    "Sleep_Duration":           "Horas de sueño",
    "Heart_Rate":               "Frecuencia cardíaca",
    "Triglycerides":            "Triglicéridos",
    "Glucose":                  "Glucosa",
    "Gender":                   "Sexo",
    "Systolic_BP":              "Presión sistólica",
    "Diastolic_BP":             "Presión diastólica",
}

_CORRECTION_THRESHOLD = {
    "Systolic_BP":             0.20,
    "Diastolic_BP":            0.20,
    "Physical_Activity_Level": 0.15,
    "Salt_Intake":             0.12,
    "Stress_Level":            0.10,
    "Sleep_Duration":          0.08,
    "BMI":                     0.08,
    "Smoking_Status":          0.06,
    "Alcohol_Intake":          0.06,
    "Age":                     0.06,
    "Triglycerides":           0.05,
    "Cholesterol":             0.05,
    "Glucose":                 0.05,
    "Family_History":          0.05,
    "Diabetes":                0.05,
    "Heart_Rate":              0.04,
    "Gender":                  0.00,
}

_SAFE_RANGE = {
    "Sleep_Duration": (6.0, 9.0),
    "Systolic_BP":    (90.0, 129.0),
    "Diastolic_BP":   (60.0, 84.0),
}

_STRESS_ZONES = {"low": (1, 3), "high": (7, 9)}

_RECOMENDACIONES_MAP = {
    "Systolic_BP":             "Control médico de la presión arterial sistólica; seguimiento periódico",
    "Diastolic_BP":            "Control médico de la presión arterial diastólica; seguimiento periódico",
    "BMI":                     "Reducir peso hacia un IMC saludable (18.5–24.9)",
    "Salt_Intake":             "Reducir ingesta de sal a menos de 2.3 g de sodio por día",
    "Physical_Activity_Level": "Incrementar actividad física de forma regular (≥150 min/semana)",
    "Stress_Level":            "Implementar técnicas de manejo del estrés (meditación, ejercicio)",
    "Sleep_Duration":          "Ajustar hábitos de sueño al rango saludable (6–9 horas)",
    "Smoking_Status":          "Abandonar el tabaco; es uno de los factores de riesgo más modificables",
    "Alcohol_Intake":          "Reducir o eliminar el consumo de alcohol",
    "Triglycerides":           "Controlar triglicéridos con dieta baja en carbohidratos refinados",
    "Cholesterol":             "Control del colesterol con dieta, ejercicio y seguimiento médico",
    "Glucose":                 "Monitorear glucemia; valores elevados aumentan el riesgo cardiovascular",
    "Family_History":          "Seguimiento médico preventivo por antecedentes familiares de HTA",
    "Diabetes":                "Control estricto de la diabetes para reducir el riesgo cardiovascular",
    "Age":                     "Control médico preventivo periódico por edad",
    "Heart_Rate":              "Monitorear frecuencia cardíaca en reposo",
}


# ─────────────────────────────────────────────
# Corrección clínica de SHAP
# ─────────────────────────────────────────────
def _clinical_effect(variable: str, valor: float, shap_impacto: float):
    """Corrige la dirección del SHAP cuando contradice la evidencia clínica y la señal es débil."""
    umbral   = _CORRECTION_THRESHOLD.get(variable, 0.0)
    abs_imp  = abs(shap_impacto)
    es_ruido = abs_imp < umbral

    if variable == "Systolic_BP":
        lo, hi   = _SAFE_RANGE["Systolic_BP"]
        en_rango = lo <= valor <= hi
        correcto = shap_impacto <= 0 if en_rango else shap_impacto >= 0
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "Diastolic_BP":
        lo, hi   = _SAFE_RANGE["Diastolic_BP"]
        en_rango = lo <= valor <= hi
        correcto = shap_impacto <= 0 if en_rango else shap_impacto >= 0
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "Physical_Activity_Level":
        if int(valor) == 2:
            correcto = shap_impacto <= 0;  umbral_local = umbral
        elif int(valor) == 1:
            correcto = shap_impacto <= 0;  umbral_local = umbral * 0.5
        else:
            correcto = shap_impacto >= 0;  umbral_local = umbral
        if not correcto and abs_imp < umbral_local:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "Smoking_Status":
        if int(valor) == 0:
            correcto = shap_impacto <= 0;  umbral_local = umbral
        elif int(valor) == 2:
            correcto = shap_impacto >= 0;  umbral_local = umbral
        else:
            return shap_impacto, False   # moderado: cualquier dirección es aceptable
        if not correcto and abs_imp < umbral_local:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "Sleep_Duration":
        lo, hi   = _SAFE_RANGE["Sleep_Duration"]
        en_rango = lo <= valor <= hi
        correcto = shap_impacto <= 0 if en_rango else shap_impacto >= 0
        if not correcto and es_ruido:
            return -shap_impacto, True
        return shap_impacto, False

    if variable == "Stress_Level":
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
        "Salt_Intake", "BMI", "Alcohol_Intake", "Age", "Triglycerides",
        "Cholesterol", "Glucose", "Family_History", "Diabetes", "Heart_Rate",
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


def _regularizar_probabilidades(pred_proba: np.ndarray) -> np.ndarray:
    """Suaviza probabilidades extremas conservando orden y suma total."""
    probabilidades = np.asarray(pred_proba, dtype=float)
    soporte = max(len(getattr(_model, "estimators_", [])), 1)
    alpha = _PROBABILITY_PRIOR_STRENGTH

    suavizadas = (probabilidades * soporte + alpha) / (
        soporte + alpha * probabilidades.size
    )
    return suavizadas / np.sum(suavizadas)


# ─────────────────────────────────────────────
# Función principal de predicción
# ─────────────────────────────────────────────
def predecir_hipertension(datos: dict) -> dict:
    """
    Predice el riesgo de hipertensión arterial de un paciente.

    Parámetros (claves del dict — mayúsculas/PascalCase tal como el modelo fue entrenado):
        Age                    float  Edad en años
        BMI                    float  IMC (kg/m²)
        Cholesterol            float  Colesterol total (mg/dL)
        Smoking_Status         int    0=No fumador  1=Ex/Ocasional  2=Fumador activo
        Alcohol_Intake         float  Consumo de alcohol (unidades/día)
                                      Mapeo sugerido desde BD (alcohol_frecuency):
                                        'never'        → 0.0
                                        'occasionally' → 0.5
                                        'weekly'       → 1.0
                                        'daily'        → 2.0
        Physical_Activity_Level int   0=Bajo  1=Moderado  2=Alto
        Family_History         int    0=No  1=Sí  ← family_history_htn (bool → 0/1)
        Diabetes               int    0=No  1=Sí  ← diabetes_diagnosed (bool → 0/1)
        Stress_Level           int    Estrés (1–9); capturar a 9 si BD devuelve 10
        Salt_Intake            float  Ingesta de sal (g/día) ← salt_intake en BD
        Sleep_Duration         float  Horas de sueño ← sleep_hours en BD
        Heart_Rate             float  Frecuencia cardíaca (lpm) ← heart_rate en BD
        Triglycerides          float  Triglicéridos (mg/dL) ← triglycerides_level en BD
        Glucose                float  Glucosa en sangre (mg/dL) ← glucose en BD
        Gender                 int    0=Mujer  1=Hombre ← sex_at_birth en BD
        Systolic_BP            float  Presión sistólica (mmHg) ← measurement.systolic_bp
        Diastolic_BP           float  Presión diastólica (mmHg) ← measurement.diastolic_bp

    Retorna dict con: puntuacion, clasificacion, nivel_riesgo,
                      probabilidades, factores_principales, recomendaciones
    """
    paciente         = pd.DataFrame([datos])[_feature_names]
    valores_paciente = paciente.values[0]

    pred_class     = int(_model.predict(paciente)[0])
    pred_proba_raw = _model.predict_proba(paciente)[0]
    pred_proba     = _regularizar_probabilidades(pred_proba_raw)

    # SHAP — referenciado a la clase predicha
    shap_values = _explainer.shap_values(paciente)
    if isinstance(shap_values, list):
        shap_raw = shap_values[pred_class][0]
    else:
        shap_raw = shap_values[0, :, pred_class]

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

    puntuacion_continua = float(pred_proba[1]) * 50 + float(pred_proba[2]) * 100
    puntuacion = int(np.clip(round(puntuacion_continua), 1, 99))

    top_factores = impacto_df[impacto_df["Variable"] != "Gender"].head(4)
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
        "puntuacion":    puntuacion,
        "clasificacion": _LABELS[pred_class],
        "nivel_riesgo":  _NIVEL_RIESGO[pred_class],
        "probabilidades": {
            "sin_hipertension": round(float(pred_proba[0]) * 100, 1),
            "prehipertension":  round(float(pred_proba[1]) * 100, 1),
            "con_hipertension": round(float(pred_proba[2]) * 100, 1),
        },
        "factores_principales": factores_ui,
        "recomendaciones":      recomendaciones,
    }
