from modules.rag.simple_rag import split_document_into_chuncks


if __name__ == "__main__":
  chunks = split_document_into_chuncks('ai.pdf')
  for chunk in chunks:
    print(chunk)
    print("\n\n" + "-"*80)



