# from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.chains.transform import TransformChain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SimpleSequentialChain
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
 
with open("../../text_files/state_of_the_union.txt") as f:
    state_of_union = f.read()

def transform_func(inputs:dict) -> dict:
    text = inputs["text"]
    short_text = "  ".join(text.split("\n")[0])
    # short_text = text.upper()
    return {"output_text": short_text}

transform_chain = TransformChain(input_variables=["text"], output_variables=["output_text"], transform=transform_func)

template = """Summarize this text:
 {output_text}
Summary:"""

prompt = PromptTemplate(input_variables=["output_text"], template=template)
llm_chain = LLMChain(llm=OpenAI(), prompt=prompt)

sequential_chain = SimpleSequentialChain(chains=[transform_chain, llm_chain])

res = sequential_chain.invoke(state_of_union)
print(res)