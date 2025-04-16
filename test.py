from modules.rag.simple_rag import Rag


if __name__ == "__main__":
  rag_instance = Rag()
  anwser = rag_instance.get_answer('cuales son los pasos para realizar el trabajo')
  print(anwser)



