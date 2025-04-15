from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

deepseek_api_key = os.getenv('DEEPSEEK_KEY')
llm = OpenAI(base_url='//api.deepseek.com', api_key=deepseek_api_key)



class Rag():
  pass