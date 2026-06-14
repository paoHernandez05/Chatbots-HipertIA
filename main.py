from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import responder_pregunta

from diabetes_predictor import predecir_diabetes
from hypertension_predictor import predecir_hipertension

app = FastAPI(
    title="HIPERGIA API",
    version="2.0.0",
    description="API de salud predictiva: RAG + modelos de riesgo cardiovascular y metabólico.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════
# Modelos de petición
# ══════════════════════════════════════════════

class PreguntaRequest(BaseModel):
    pregunta: str


class DiabetesRequest(BaseModel):
    """
    Campos para el modelo de riesgo de diabetes.
    Mapeo desde la BD (tabla profiles):
        age                        ← calcular desde dob
        gender                     ← sex_at_birth ('female'→0, 'male'→1)
        bmi                        ← bmi
        blood_pressure             ← measurement->>'systolic_bp'  (presión sistólica mmHg)
        insulin_level              ← insulin_level
        cholesterol_level          ← cholesterol_level
        triglycerides_level        ← triglycerides_level
        physical_activity_level    ← physical_activity_level  (0/1/2)
        daily_calorie_intake       ← daily_calorie_intake
        sugar_intake_grams_per_day ← sugar_intake_grams_per_day
        sleep_hours                ← sleep_hours
        stress_level               ← stress_level  (1–10)
        family_history_diabetes    ← family_history_diabetes  (0/1)
        waist_circumference_cm     ← waist_circumference_cm
    """
    age: float                           = Field(..., description="Edad en años")
    gender: int                          = Field(..., ge=0, le=1,
                                                  description="0=Femenino  1=Masculino")
    bmi: float                           = Field(..., description="IMC (kg/m²)")
    blood_pressure: float                = Field(..., description="Presión sistólica (mmHg)")
    insulin_level: float                 = Field(..., description="Insulina (µU/mL)")
    cholesterol_level: float             = Field(..., description="Colesterol total (mg/dL)")
    triglycerides_level: float           = Field(..., description="Triglicéridos (mg/dL)")
    physical_activity_level: int         = Field(..., ge=0, le=2,
                                                  description="0=Sin actividad  1=Algo semanal  2=Constante")
    daily_calorie_intake: float          = Field(..., description="Calorías diarias (kcal)")
    sugar_intake_grams_per_day: float    = Field(..., description="Azúcar (g/día)")
    sleep_hours: float                   = Field(..., description="Horas de sueño")
    stress_level: int                    = Field(..., ge=1, le=10, description="Estrés (1–10)")
    family_history_diabetes: int         = Field(..., ge=0, le=1, description="0=No  1=Sí")
    waist_circumference_cm: float        = Field(..., description="Circunferencia de cintura (cm)")


class HypertensionRequest(BaseModel):
    """
    Campos para el modelo de riesgo de hipertensión.
    Mapeo desde la BD (tabla profiles):
        Age                    ← calcular desde dob
        BMI                    ← bmi
        Cholesterol            ← cholesterol_level
        Smoking_Status         ← smoker_status  (0/1/2)
        Alcohol_Intake         ← alcohol_frecuency (texto → numérico; ver descripción)
        Physical_Activity_Level← physical_activity_level  (0/1/2)
        Family_History         ← family_history_htn  (bool → 0/1)
        Diabetes               ← diabetes_diagnosed  (bool → 0/1)
        Stress_Level           ← stress_level  (1–9; capturar a 9 si BD devuelve 10)
        Salt_Intake            ← salt_intake
        Sleep_Duration         ← sleep_hours
        Heart_Rate             ← heart_rate
        Triglycerides          ← triglycerides_level
        Glucose                ← glucose
        Gender                 ← sex_at_birth  ('female'→0, 'male'→1)
        Systolic_BP            ← measurement->>'systolic_bp'
        Diastolic_BP           ← measurement->>'diastolic_bp'

    Mapeo sugerido para alcohol_frecuency (texto → float):
        'never'        → 0.0
        'occasionally' → 0.5
        'weekly'       → 1.0
        'daily'        → 2.0
        'heavy'        → 3.0
    """
    Age: float                           = Field(..., description="Edad en años")
    BMI: float                           = Field(..., description="IMC (kg/m²)")
    Cholesterol: float                   = Field(..., description="Colesterol total (mg/dL)")
    Smoking_Status: int                  = Field(..., ge=0, le=2,
                                                  description="0=No fumador  1=Ex/Ocasional  2=Fumador activo")
    Alcohol_Intake: float                = Field(..., description="Consumo de alcohol (unidades/día)")
    Physical_Activity_Level: int         = Field(..., ge=0, le=2,
                                                  description="0=Bajo  1=Moderado  2=Alto")
    Family_History: int                  = Field(..., ge=0, le=1,
                                                  description="0=No  1=Sí  (family_history_htn)")
    Diabetes: int                        = Field(..., ge=0, le=1,
                                                  description="0=No  1=Sí  (diabetes_diagnosed)")
    Stress_Level: int                    = Field(..., ge=1, le=9, description="Estrés (1–9)")
    Salt_Intake: float                   = Field(..., description="Ingesta de sal (g/día)")
    Sleep_Duration: float                = Field(..., description="Horas de sueño")
    Heart_Rate: float                    = Field(..., description="Frecuencia cardíaca (lpm)")
    Triglycerides: float                 = Field(..., description="Triglicéridos (mg/dL)")
    Glucose: float                       = Field(..., description="Glucosa en sangre (mg/dL)")
    Gender: int                          = Field(..., ge=0, le=1, description="0=Mujer  1=Hombre")
    Systolic_BP: float                   = Field(..., description="Presión sistólica (mmHg)")
    Diastolic_BP: float                  = Field(..., description="Presión diastólica (mmHg)")


# ══════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "version": app.version}


@app.post("/preguntar")
def preguntar(data: PreguntaRequest):
    try:
        respuesta = responder_pregunta(data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predecir/diabetes")
def endpoint_diabetes(data: DiabetesRequest):
    """
    Predice el riesgo de diabetes (Bajo Riesgo / Prediabetes / Alto Riesgo)
    y devuelve factores SHAP con corrección clínica.
    """
    try:
        resultado = predecir_diabetes(data.model_dump())
        return resultado
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature no encontrado en el modelo: {e}. "
                   "Verifica que los nombres de campo coincidan con los del .pkl entrenado.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predecir/hipertension")
def endpoint_hipertension(data: HypertensionRequest):
    """
    Predice el riesgo de hipertensión (Sin Hipertensión / Prehipertensión / Con Hipertensión)
    y devuelve factores SHAP con corrección clínica.
    """
    try:
        resultado = predecir_hipertension(data.model_dump())
        return resultado
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature no encontrado en el modelo: {e}. "
                   "Verifica que los nombres de campo coincidan con los del .pkl entrenado.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))