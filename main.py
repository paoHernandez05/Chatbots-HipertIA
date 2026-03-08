from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import responder_pregunta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes limitar esto a ["http://127.0.0.1:5500"] si gustas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreguntaRequest(BaseModel):
    pregunta: str

@app.post("/preguntar")
def preguntar(data: PreguntaRequest):
    try:
        respuesta = responder_pregunta(data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

