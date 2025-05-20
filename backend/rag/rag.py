from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


def create_vector_db(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PDFPlumberLoader
    )
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        #250 - 50
        chunk_size=200,
        chunk_overlap=30
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"}
    )
    
    vector_db = FAISS.from_documents(texts, embeddings)
    return vector_db

# def get_context(vector_db, keywords):
#     context = ""
#     for key, values in keywords.items():
#         # If values is a list, process each value
#         if isinstance(values, list):
#             for value in values:
#                 if value:  # Check if value is not empty
#                     docs = vector_db.similarity_search(value, 1)
#                     for doc in docs:
#                         context += f"\n\n{doc.page_content}"
#         # If values is a string or another type, process it directly
#         elif values:
#             docs = vector_db.similarity_search(str(values), 1)
#             for doc in docs:
#                 context += f"\n\n{doc.page_content}"
    
#     return context

def get_context(vector_db, keywords, max_sentences=3):
    """
    Get context from vector database with a maximum of 3 sentences per context type.
    
    Args:
        vector_db: The vector database to search
        keywords: Dictionary of keywords
        max_sentences: Maximum number of sentences to include (default: 3)
    
    Returns:
        str: Retrieved context limited to max_sentences
    """
    context = ""
    total_sentences = 0
    
    for key, values in keywords.items():
        # If we've already hit our sentence limit, break
        if total_sentences >= max_sentences:
            break
            
        # If values is a list, process each value
        if isinstance(values, list):
            for value in values:
                if value and total_sentences < max_sentences:  # Check if value is not empty
                    docs = vector_db.similarity_search(value, 1)
                    for doc in docs:
                        # Split the document content into sentences
                        # Simple split by period followed by space or end of string
                        sentences = [s.strip() + "." for s in doc.page_content.split(".") if s.strip()]
                        
                        # Take only enough sentences to reach our max
                        sentences_to_add = sentences[:max_sentences - total_sentences]
                        
                        if sentences_to_add:
                            context += f"\n\n{' '.join(sentences_to_add)}"
                            total_sentences += len(sentences_to_add)
                        
                        # If we've reached our limit, stop adding more
                        if total_sentences >= max_sentences:
                            break
        
        # If values is a string or another type, process it directly
        elif values and total_sentences < max_sentences:
            docs = vector_db.similarity_search(str(values), 1)  # Reduced from 2 to 1 for brevity
            for doc in docs:
                # Split into sentences
                sentences = [s.strip() + "." for s in doc.page_content.split(".") if s.strip()]
                
                # Take only enough sentences to reach our max
                sentences_to_add = sentences[:max_sentences - total_sentences]
                
                if sentences_to_add:
                    context += f"\n\n{' '.join(sentences_to_add)}"
                    total_sentences += len(sentences_to_add)
                
                # If we've reached our limit, stop adding more
                if total_sentences >= max_sentences:
                    break
    
    return context

def rag_pipeline(llm, query, keywords, vector_db_folklore, vector_db_scientific, prompt, stream=False):
    if isinstance(prompt, list):
        prompt = "\n".join(prompt)  # Join list into a single string if necessary
    prompt_template = PromptTemplate(
        input_variables=["context", "query","keywords"],
        template=prompt
    )
    list_keywords = [f"{key}: {val}" for key, val in keywords.items()]
    print("List keywords: ", list_keywords)
    keywords_string = ""
    for kw in list_keywords:
        keywords_string += kw
        keywords_string += "\n"
    print("Keywords string: ", keywords_string)
    context_folklore = get_context(vector_db_folklore, keywords)
    context_scientific = get_context(vector_db_scientific, keywords)
    context = "Folklore context: " + context_folklore + "\n" + "Scientific context:" + context_scientific
    formatted_prompt = prompt_template.format(context=context, query=query, keywords=keywords)
    messages = [
        HumanMessage(content=formatted_prompt)
    ]
    
    if stream:
        return llm.stream(messages)
    else:
        response = llm(messages)
        return response