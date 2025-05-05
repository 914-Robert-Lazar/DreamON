from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PDFMinerLoader
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_db(directory_path: str) -> FAISS:
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PDFMinerLoader
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db

def get_context(vector_db,keywords):
    context = ""
    for kw in keywords:
        docs = vector_db.similarity_search(kw,2)
        context = context.join(["\n\n",doc.page_content for doc in docs])

    return context

def rag_pipeline(llm,query,keywords,vector_db,prompt) -> str:
    prompt_template = PromptTemplate(
        input_variables=["context", "query","keywords"],
        template=prompt
    )
    list_keywords = res = [f"{key}: {val}" for key, val in keywords.items()]
    keywords_string = ""
    for kw in list_keywords:
        keywords_string+=kw
        keywords_string+="\n"
    context = get_context(vector_db,keywords)
    formatted_prompt = prompt_template.format(context=context, query=query,keywords=keywords)
    response = llm(formatted_prompt)
    
    return response
