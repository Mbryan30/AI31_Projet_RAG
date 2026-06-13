"""
Point d'entrée alternatif — depuis la racine backend/ :
    python run.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,          # hot-reload en dev
        log_level="info",
    )
