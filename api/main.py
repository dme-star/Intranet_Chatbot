import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import uvicorn
from embedding_store import EmbeddingStore
from fastapi.responses import HTMLResponse, FileResponse

INTRANET_DATA_DIR = os.environ.get("INTRANET_DATA_DIR", "/data")
os.makedirs(INTRANET_DATA_DIR, exist_ok=True)

app = FastAPI(title="Intranet Chatbot API (Simple)")

store = EmbeddingStore(data_dir=INTRANET_DATA_DIR)

class Query(BaseModel):
    query: str

@app.on_event("startup")
async def startup():
    store.load()

@app.post("/ingest/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode('utf-8')
    except:
        text = None
    if text is None:
        raise HTTPException(status_code=400, detail="Only text files (.txt) supported in this simple demo.")
    path = os.path.join(INTRANET_DATA_DIR, file.filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    store.ingest_file(path)
    return {"status":"ok", "filename": file.filename}

@app.post("/chat")
def chat(q: Query):
    results = store.query(q.query, top_k=5)
    answer = "\n\n".join([f"Source: {r['source']}\n{r['text']}" for r in results])
    return {"answer": answer, "results": results}

@app.get("/widget", response_class=HTMLResponse)
def widget():
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "frontend", "widget.html"))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
