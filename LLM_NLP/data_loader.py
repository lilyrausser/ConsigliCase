from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdfs(pdf_paths):
    ''' loads and splits all PDFs into smaller text chunks'''
    
    docs = []
    # defines how text should be segmented into smaller chunks (overlapping as well)
    text_splitter = RecursiveCharacterTextSplitter(
    # max characters per chunk
    chunk_size=600,
    # overlap between consecutive chunks (maintain continuity)
    chunk_overlap=100,
    # focus on splitting at logical text boundaries 
    separators=["\n\n", "\n", ". ", " "]
    )
    # iterate through each PDF file 
    for pdf in pdf_paths:
        # extract text from the pdf 
        loader = PyPDFLoader(pdf)
        # list of document objects, each representing a page
        pages = loader.load()
        # chunk each page into small sections
        split_docs = text_splitter.split_documents(pages)
        # collect all chunked documents to a single list
        docs.extend(split_docs)
    return docs
  

def clean_docs(docs):
    '''removes empty of very short chunks of text that do not provide meaningful information'''
    cleaned_docs = [doc for doc in docs if len(doc.page_content.strip()) > 20]
    return cleaned_docs


