from fastapi import FastAPI

from routes.register import router as register_router
from routes.verify import router as verify_router

app = FastAPI(
    title="Face AI Service",
)

app.include_router(register_router)
app.include_router(verify_router)


@app.get("/")
def root():
    return {
        "message": "Face AI Service"
    }
