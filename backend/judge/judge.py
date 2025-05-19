from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

def separate_first_line(response: str) -> tuple[str, str]:
    """
    Separates the first line from the rest of the response string.
    
    Args:
        response (str): The input response string
        
    Returns:
        tuple[str, str]: A tuple containing (first_line, remaining_text)
    """
    lines = response.strip().split('\n', 1)
    first_line = lines[0]
    remaining_text = lines[1] if len(lines) > 1 else ""
    return first_line, remaining_text

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
    first_line,reason = separate_first_line(response)
    valid = False
    if "Valid" in first_line:
        valid = True
    return valid,reason
