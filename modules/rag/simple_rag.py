from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from sentence_transformers import SentenceTransformer
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
import lancedb
from lancedb.pydantic import LanceModel
from pydantic import BaseModel
from langfuse import Langfuse

import os

load_dotenv()

deepseek_api_key = os.getenv('DEEPSEEK_KEY')

llm = OpenAI(base_url='https://api.deepseek.com', api_key=deepseek_api_key)

langfuse = Langfuse(
  secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
  public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
  host=os.getenv('LANGFUSE_HOST')
)

func = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2", device="cpu")

class Schema(LanceModel):
      id: int
      text: str = func.SourceField()
      vector: func.VectorField()

      class Config:
          arbitrary_types_allowed = True

class retrieval:
  def __init__(self):
     pass
   
  def embeddings(self, chunks):

      # convert the chunks into text strings
      chunks_text = [chunk.text for chunk in chunks if hasattr(chunk, "text")]
      model = SentenceTransformer("all-MiniLM-L6-v2")
      embs = model.encode(chunks_text, convert_to_numpy=True)
      embedding_dim = embs.shape[1]
      
      # insert to the vector store
      db = lancedb.connect('vector_store/db')
      table_name = 'documment_chunks'
      
      docs = []
      
      # check if the connection exits
      
      for i, (text, emb) in enumerate(zip(chunks_text, embs)):
          docs.append({
              "id": i,
              "text": text,
              "vector": emb.tolist()  # Convert to numpy
          })
      
      #insert to the database
      
      
      try:
        collection = db.create_table(table_name, docs)
      except Exception as e:
        print(e)
        collection = db.open_table(table_name)
        collection.add(data=docs)
        
      return(f"Inserted {len(docs)} documents in the collection '{table_name}'.")


  def load_documents(self, pdf):
    elements = partition_pdf(
      pdf,
      strategy="auto",
      infer_table_structure=True,
      include_page_breaks=False)
    return elements

  def split_document_into_chuncks(self, document):
    chunks = chunk_elements(document)
    return chunks

  def process_document(self, pdf):
    
    #process the document with unstructured
    document = self.load_documents(pdf)
    
    #split the document into chunks
    chunks = self.split_document_into_chuncks(document)
    
    # embeds the chunks and insert them to vector store
    embs = self.embeddings(chunks)
    return 'the document was inserted into the vector store' + embs
    
    

class Rag:
    def __init__(self):
        pass

    def get_context(self, query):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_query = model.encode([query], convert_to_numpy=True)
        
        db = lancedb.connect('vector_store/db')
        collection = db.open_table('documment_chunks')
        
        docs = collection.search(emb_query).limit(3).to_pydantic(Schema)
        
        context = ''
        i = 1
        for doc in docs:
            context += f'chunk number {i} \n'
            context += f'{doc.text} \n'
            i += 1
        return context

    def get_answer(self, question):
        trace = langfuse.trace(name=f'chain execution => {question}', input={'user': question})
        generation = trace.generation(name = 'llm_generation', input = {'user': question})
      
      
        context = self.get_context(question)
        
        answer = llm.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": f"Here is the question from the user: {question}"}
            ]
        )
        
        response = answer.choices[0].message.content
        
        generation.end(
          output = response,
          metadata = {'context': context}
        )
        
        
        return response
