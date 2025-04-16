from fastapi import FastAPI, UploadFile, File, HTTPException
from modules.rag.simple_rag import retrieval, Rag
import os
import uuid
import shutil

rag_instance = None

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    # save the file temp
    file_id = str(uuid.uuid4())
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, f"{file_id}.pdf")
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # process document, extract text, create embedding and save to the vector store
    process = retrieval()
    document = process.process_document(temp_file_path)
    
    
    os.remove(temp_file_path)
    return {"detail": f"{document}"}
  
@app.post('/question')
async def return_answer(question: str):
  rag_instance = Rag()
  answer = rag_instance.get_answer(question)
  return {"answer": answer}