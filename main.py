from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json
from transformers import pipeline

app = FastAPI()

# ============================
# 1. Cargar embeddings y FAISS
# ============================
MODEL_NAME = "intfloat/multilingual-e5-large"
embedder = SentenceTransformer(MODEL_NAME)

index = faiss.read_index("five9_faiss.index")
with open("five9_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"🧠 Sistema cargado: {len(chunks)} chunks, FAISS {index.ntotal} vectores")

# ============================
# 2. Cargar modelo LLM ligero
# ============================
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",  # modelo seguro para CPU
    device=-1,                        # CPU
    max_new_tokens=128,               # límite seguro para Render
    temperature=0.3,
)

# ============================
# 3. Modelo de request
# ============================
class Query(BaseModel):
    question: str

# ============================
# 4. Función de búsqueda en FAISS
# ============================
def retrieve_chunks(question, k=5):
    q_emb = embedder.encode([question])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

# ============================
# 5. Endpoint de Q&A
# ============================
@app.post("/query")
async def query_five9(data: Query):
    question = data.question.strip()
    print(f"\n🔍 Pregunta: {question}")

    # Recuperar contexto
    relevant_chunks = retrieve_chunks(question, k=5)
    if not relevant_chunks:
        return {"answer": f"No encontrado en la guía para '{question}'", "context": ""}

    context = "\n\n".join(relevant_chunks)

    # Construir prompt estricto
    prompt = f"""
You are a technical assistant specialized in Five9 documentation.  
Answer the question ONLY if the provided context explicitly contains relevant information.  

Rules:
- If the context does not explain the requested action, respond with: "Not found in the guide."  
- Do not infer, guess, or use external knowledge.  
- Focus ONLY on instructions, steps, or procedures in the context.  

Question: {question}

Context:
{context}

Answer:
"""

    # Generar respuesta (CPU, seguro en Render)
    response = generator(prompt, do_sample=False, max_new_tokens=128)[0]["generated_text"]

    return {
        "answer": response.strip(),
        "context": context[:1000] + "..." if len(context) > 1000 else context
    }

# ============================
# 6. Ejecutar local (para pruebas)
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
