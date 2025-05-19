from rag.rag import rag_pipeline,create_vector_db
from keyword_extraction.gliner_keyword_extraction import keyword_extraction_gliner,model_gliner
# from keyword_extraction.spacy_keyword_extraction import keyword_extraction_spacy,model_spacy
from langchain_openai import ChatOpenAI
from judge.judge import judge
from judge.redo_rag_output import redo_rag_output

pdf_path_folklore = "../Folklore articles/"
pdf_path_scientific = "../Scientific articles/"
vector_db_folklore = create_vector_db(pdf_path_folklore)
vector_db_scientific = create_vector_db(pdf_path_scientific)
prompt_path = "../prompt/prompt.txt"
judge_prompt_path = "../prompt/judge.txt"
redo_rag_prompt_path = "../prompt/redo_rag.txt"
def pipeline(query, stream=False):    
    base_url = "http://localhost:1234/v1"
    api_key = "lm-studio"
    llm_model = "meta-llama_-_meta-llama-3-8b"
    llm_model_judge = "gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf"
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        temperature=0.4,
        model=llm_model,
        streaming=stream
    )
    llm_judge = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        temperature=0.01,
        model=llm_model_judge
    )
    # keywords = keyword_extraction_spacy(query,model_spacy)
    keywords = keyword_extraction_gliner(query,model_gliner)
    
    prompt = ""
    with open(prompt_path,"r") as f:
        prompt = f.readlines()
        
    judge_prompt = ""
    with open(judge_prompt_path,"r") as f:
        judge_prompt = f.readlines()
        
    redo_rag_prompt = ""
    with open(redo_rag_prompt_path,"r") as f:
        redo_rag_prompt = f.readlines()
    
    if stream:
        return rag_pipeline(llm, query, keywords, vector_db_folklore, vector_db_scientific, prompt, stream=True)
    else:
        output = rag_pipeline(llm, query, keywords, vector_db_folklore, vector_db_scientific, prompt)
        output_ok = False
        count = 0
        while not output_ok and count < 3:
            output_ok, judge_output = judge(llm_judge, output, judge_prompt)
            if not output_ok:
                count += 1
                output = redo_rag_output(llm, judge_output, output, redo_rag_prompt)
        if not output_ok:
            output = "Could not generate response"
        return output
        return output