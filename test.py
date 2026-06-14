import requests, json
from datetime import date

BASE_URL = "http://localhost:8000"  

# ── helpers ────────────────────────────────────────────────

def calcular_edad(dob: date) -> int:
    hoy = date.today()
    return hoy.year - dob.year - ((hoy.month, hoy.day) < (dob.month, dob.day))

def alcohol_a_numero(frecuencia: str) -> float:
    mapa = {"never": 0.0, "occasionally": 0.5,
            "weekly": 1.0, "daily": 2.0, "heavy": 3.0}
    return mapa.get(frecuencia.lower(), 0.0)

def post_prediccion(endpoint: str, payload: dict) -> dict:
    url = f"{BASE_URL}{endpoint}"
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


perfil = {
    "dob": date(1979, 6, 15),
    "sex_at_birth": "male",
    "bmi": 28.5,
    "cholesterol_level": 210.0,
    "triglycerides_level": 180.0,
    "physical_activity_level": 1,
    "daily_calorie_intake": 2200,
    "sugar_intake_grams_per_day": 55.0,
    "sleep_hours": 6.5,
    "stress_level": 7,
    "family_history_diabetes": 1,
    "waist_circumference_cm": 95.0,
    "insulin_level": 18.0,
    "smoker_status": 1,
    "alcohol_frecuency": "weekly",
    "family_history_htn": True,
    "diabetes_diagnosed": False,
    "salt_intake": 4.5,
    "heart_rate": 88,
    "glucose": 105,
    "measurement": {"systolic_bp": 148, "diastolic_bp": 94},
}


edad   = calcular_edad(perfil["dob"])
genero = 1 if perfil["sex_at_birth"] == "male" else 0

payload_diabetes = {
    "age":                        edad,
    "gender":                     genero,
    "bmi":                        perfil["bmi"],
    "blood_pressure":             float(perfil["measurement"]["systolic_bp"]),
    "insulin_level":              perfil["insulin_level"],
    "cholesterol_level":          perfil["cholesterol_level"],
    "triglycerides_level":        perfil["triglycerides_level"],
    "physical_activity_level":    perfil["physical_activity_level"],
    "daily_calorie_intake":        perfil["daily_calorie_intake"],
    "sugar_intake_grams_per_day":  perfil["sugar_intake_grams_per_day"],
    "sleep_hours":                 perfil["sleep_hours"],
    "stress_level":               perfil["stress_level"],
    "family_history_diabetes":    perfil["family_history_diabetes"],
    "waist_circumference_cm":     perfil["waist_circumference_cm"],
}

payload_htn = {
    "Age":                     edad,
    "BMI":                     perfil["bmi"],
    "Cholesterol":             perfil["cholesterol_level"],
    "Smoking_Status":          perfil["smoker_status"],
    "Alcohol_Intake":          alcohol_a_numero(perfil["alcohol_frecuency"]),
    "Physical_Activity_Level": perfil["physical_activity_level"],
    "Family_History":          int(perfil["family_history_htn"]),
    "Diabetes":                int(perfil["diabetes_diagnosed"]),
    "Stress_Level":            min(perfil["stress_level"], 9),  # cap a 9
    "Salt_Intake":             perfil["salt_intake"],
    "Sleep_Duration":          perfil["sleep_hours"],
    "Heart_Rate":              float(perfil["heart_rate"]),
    "Triglycerides":           perfil["triglycerides_level"],
    "Glucose":                 float(perfil["glucose"]),
    "Gender":                  genero,
    "Systolic_BP":             float(perfil["measurement"]["systolic_bp"]),
    "Diastolic_BP":            float(perfil["measurement"]["diastolic_bp"]),
}


if __name__ == "__main__":
    print("=" * 55)
    print("  DIABETES")
    print("=" * 55)
    res_db = post_prediccion("/predecir/diabetes", payload_diabetes)
    print(json.dumps(res_db, indent=2, ensure_ascii=False))

    print()
    print("=" * 55)
    print("  HIPERTENSIÓN")
    print("=" * 55)
    res_htn = post_prediccion("/predecir/hipertension", payload_htn)
    print(json.dumps(res_htn, indent=2, ensure_ascii=False))