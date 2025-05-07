from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from pipeline import pipeline

app = FastAPI(title="DreamON API", description="API for dream analysis and interpretation")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Dream(BaseModel):
    id: Optional[int] = None
    title: str
    content: str
    date: str
    keywords: Optional[List[str]] = []
    interpretation: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str

# In-memory database for demo purposes
dreams_db = []
dream_id_counter = 1

# Routes
@app.get("/")
async def root():
    return {"message": "Welcome to the DreamON API!"}

@app.get("/dreams", response_model=List[Dream])
async def get_dreams():
    return dreams_db

@app.get("/dreams/{dream_id}", response_model=Dream)
async def get_dream(dream_id: int):
    for dream in dreams_db:
        if dream.id == dream_id:
            return dream
    raise HTTPException(status_code=404, detail="Dream not found")

@app.post("/dreams", response_model=Dream)
async def create_dream(dream: Dream):
    global dream_id_counter
    dream.id = dream_id_counter
    dream_id_counter += 1
    dreams_db.append(dream)
    return dream

@app.put("/dreams/{dream_id}", response_model=Dream)
async def update_dream(dream_id: int, updated_dream: Dream):
    for i, dream in enumerate(dreams_db):
        if dream.id == dream_id:
            updated_dream.id = dream_id
            dreams_db[i] = updated_dream
            return updated_dream
    raise HTTPException(status_code=404, detail="Dream not found")

@app.delete("/dreams/{dream_id}")
async def delete_dream(dream_id: int):
    for i, dream in enumerate(dreams_db):
        if dream.id == dream_id:
            dreams_db.pop(i)
            return {"message": "Dream deleted successfully"}
    raise HTTPException(status_code=404, detail="Dream not found")

@app.post("/process_query")
async def process_query(request: QueryRequest):
    result = pipeline(request.query)
    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)