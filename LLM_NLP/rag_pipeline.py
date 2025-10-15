import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_rag_pipeline(docs):
    '''builds rag pipleine given the documents'''
    
    # embed the documents 
    embeddings = OpenAIEmbeddings()

    # store the embeddings in Chroma (running locally), batched to avoid memory issues when running so many chunks 
    batch_size = 5000
    db = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if db is None:
            db = Chroma.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    
    # retriever to get the most relevant document chunks 
    retriever = db.as_retriever(search_kwargs={"k":6})
    # initialize LLM 
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    
    # defining concise prompt to aid the model 
    template = '''
    You are a precise financial analyst.
    Answer the question strictly using the provided context.
    - Quote the exact number and units as written.
    - Do not add reasoning, speculation, or extra sentences.
    - If the exact answer is not found, reply: "I don't know."

    Context: {context}
    Question: {question}
    Answer (one short factual sentence):'''

    prompt = PromptTemplate(
        input_variables=['context', 'question'],
        template=template
    )
    
    # combine retriever and LLM into a single RetrievalQA chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,
        # simplest method to concatenate retrieved chunks to one context
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return qa_chain
