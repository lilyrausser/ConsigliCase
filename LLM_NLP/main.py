#%%
# helps load PDFs, create embeddings, store vectors, set up RAG
from data_loader import load_pdfs, clean_docs
from rag_pipeline import build_rag_pipeline
import os


#%%
def main():
    '''Main function to run the RAG pipeline'''

    # pdfs = [
    # # BMW
    # "Data/BMW/BMW_Annual_Report_2021.pdf",
    # "Data/BMW/BMW_Annual_Report_2022.pdf",
    # "Data/BMW/BMW_Annual_Report_2023.pdf",

    # # Ford 
    # "Data/Ford/Ford_Annual_Report_2021.pdf",
    # "Data/Ford/Ford_Annual_Report_2022.pdf",
    # "Data/Ford/Ford_Annual_Report_2023.pdf",

    # # Tesla 
    # "Data/Tesla/Tesla_Annual_Report_2022.pdf",
    # "Data/Tesla/Tesla_Annual_Report_2023.pdf"
    # ]   

    # define the base directoreis for each company's PDF reports 
    base_dirs = ["Data/BMW", "Data/Ford", "Data/Tesla"]

    # collect all individaul PDF file paths  
    pdfs = []
    for d in base_dirs:
        for fname in os.listdir(d):
            if fname.endswith(".pdf"):
                pdfs.append(os.path.join(d, fname))
   
    # load and clean PDF text chunks 
    docs = load_pdfs(pdfs)
    docs = clean_docs(docs)  

    # build the RAG pipeline 
    qa_chain = build_rag_pipeline(docs)
    
    # interactive loop to ask multiple questions 
    while True: 
        question = input("Ask a question (type exit or quit to leave the program): ").strip()
        if question.lower() in ['exit', 'quit']:
            break
        answer = qa_chain.run(question)
        print(f"Answer: {answer}\n")

#%%
if __name__ == "__main__":
    main()