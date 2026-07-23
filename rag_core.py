from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import os

# Carga las variables desde .env
load_dotenv()

# Obtiene la clave
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Información base de presión arterial y glucosa
with open("documents.txt", "r", encoding="utf-8") as f:
    gen_info = f.read()


# División en chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

#Chunks para presión arterial y glucosa
chunks = splitter.split_documents([Document(page_content=gen_info)])



# Modelo
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Template para presión arterial y glucosa
template = """Eres un asistente médico que responde preguntas sobre:

- glucosa en sangre
- diabetes
- presión arterial
- hipertensión

Los datos incluidos en "Datos del paciente" provienen de la cuenta del usuario y son
la fuente de verdad para preguntas personales. Si existen mediciones, debes leerlas,
indicar sus valores, unidades y fechas disponibles, y responder con base en ellas.
No afirmes que no tienes acceso a los datos cuando las listas contengan registros.
No inventes valores faltantes ni diagnósticos. Distingue información educativa de un
diagnóstico médico y recomienda atención profesional cuando corresponda.
Responde de forma clara y breve usando únicamente el contexto y los datos proporcionados.
Si la pregunta no está relacionada con estos temas, indícalo amablemente.
"""

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "{template}"),
    (
        "human",
        """Contexto educativo:
{context}

Datos del paciente (JSON):
{datos}

Pregunta actual:
{input}""",
    ),
])


# Cadena RAG
chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

def responder_pregunta(
    pregunta: str,
    perfil: dict,
    historial_presion: list[dict] | None = None,
    historial_glucosa: list[dict] | None = None,
    historial_conversacion: list[dict] | None = None,
) -> str:
    datos = json.dumps(
        {
            "perfil": perfil,
            "historialPresion": historial_presion or [],
            "historialGlucosa": historial_glucosa or [],
            "historialConversacion": historial_conversacion or [],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    respuesta = chain.invoke({
        "template": template,
        "input": pregunta,
        "context": chunks,
        "datos": datos
    })
    return respuesta
