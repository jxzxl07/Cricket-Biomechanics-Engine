from fastapi import FastAPI

app = FastAPI(title="Cricket Biomechanics API")

@app.get("/health")
def health():
    return {"status": "ok"}