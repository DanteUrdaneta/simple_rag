from modules.rag.simple_rag import load_documents


if __name__ == "__main__":
  elements = load_documents('ai.pdf')
  complete_text = ''
  for element in elements:
    complete_text += element.text
  print(complete_text)  



