from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

def judge(llm_judge,output,judge_prompt):
    judge_prompt_template = PromptTemplate(
        input_variables = ["output"],
        template = judge_prompt
    )
    formatted_prompt = judge_prompt_template.format(output=output)
    messages = [
        HumanMessage(content=formatted_prompt)
    ]
    response = llm_judge(messages)
    