from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Stuttering AI System",
    description="Backend API for Stuttering AI System",
    version="1.0.0"
)


# Health check route
@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Simple health check endpoint.
    Returns HTTP 200 with status 'ok'.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
