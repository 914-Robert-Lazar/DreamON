from rag.rag import rag_pipeline,create_vector_db
from keyword_extraction.gliner_keyword_extraction import keyword_extraction_gliner,model_gliner
# from keyword_extraction.spacy_keyword_extraction import keyword_extraction_spacy,model_spacy
from langchain_openai import ChatOpenAI

pdf_path_folklore = "../Folklore articles/"
pdf_path_scientific = "../Scientific articles/"
vector_db_folklore = create_vector_db(pdf_path_folklore)
vector_db_scientific = create_vector_db(pdf_path_scientific)
prompt_path = "../prompt/prompt.txt"
def pipeline(query):    
    base_url = "http://localhost:1234/v1"
    api_key = "lm-studio"
    llm_model = "meta-llama_-_meta-llama-3-8b"

    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        temperature=0.4,
        model=llm_model
    )
    # keywords = keyword_extraction_spacy(query,model_spacy)
    keywords = keyword_extraction_gliner(query,model_gliner)
    prompt = ""
    with open(prompt_path,"r") as f:
        prompt = f.readlines()
    output = rag_pipeline(llm,query,keywords,vector_db_folklore,vector_db_scientific,prompt)
    return output