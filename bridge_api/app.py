
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#   from fastapi.staticfiles import StaticFiles
#   from fastapi.responses import FileResponse

# Import your existing RAG FastAPI app
from rag_api.app import app as rag_app

# Create the “bridge” FastAPI app
app = FastAPI(
    title="RAG Plugin",
    version="1.0",
    openapi_url="/openapi.json",   # expose OpenAPI schema here
    docs_url=None,
    redoc_url=None,
)

# Enable CORS so Open WebUI (or any client) can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount your RAG API under /api
app.mount("/api", rag_app)

# Serve the AI-plugin manifest
@app.get("/.well-known/ai-plugin.json", include_in_schema=False)
async def plugin_manifest():
    host = os.getenv("BRIDGE_URL", "http://bridge-api:5055")
    return {
        "schema_version":"v1",
        # …
        "api": {
            "type":"openapi",
            # change here ↓
            "url": f"{host}/openapi.json",
            "is_user_authenticated": False
        },
        # …
    }


# Serve a static logo.png (put your logo at bridge-api/static/logo.png)
#   app.mount("/static", StaticFiles(directory="static"), name="static")

#@app.get("/logo.png", include_in_schema=False)
#async def logo():
 #   return FileResponse("static/logo.png")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5055"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

from pydantic import BaseModel
import requests
from fastapi import HTTPException

class RAGQuery(BaseModel):
    query: str

@app.post("/rag-answer")
def rag_answer_proxy(body: RAGQuery):
    try:
        resp = requests.post(
            "http://erciyes-rag-api:7000/rag-answer",
            json={"query": body.query},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"RAG API unreachable: {e}")
