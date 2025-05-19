from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

def redo_rag_output(llm,judge_output,output,redo_rag_prompt):
    redo_rag_prompt_template = PromptTemplate(
        input_variables = ["judge_output","output"],
        template = redo_rag_prompt
    )
    formatted_prompt = redo_rag_prompt_template.format(judge_output=judge_output,output=output)
    messages = [
        HumanMessage(content=formatted_prompt)
    ]
    response = llm(messages)
    return response