from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

import os

load_dotenv()

deepseek_api_key = os.getenv('DEEPSEEK_KEY')
llm = OpenAI(base_url='//api.deepseek.com', api_key=deepseek_api_key)

def load_documents(pdf):
  elements = partition_pdf(
    pdf,
    strategy="auto",
    infer_table_structure=True,
    include_page_breaks=False)
  return elements

class Rag():
  pass