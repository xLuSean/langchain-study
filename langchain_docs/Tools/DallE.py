from dotenv import load_dotenv
load_dotenv()

from langchain.chains.llm import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["image_desc"],
    template="Generate a detailed prompt to generate an image based on the following description{image_desc}. within 50 words",
)
chain = LLMChain(llm=llm, prompt=prompt)
image_prompt = chain.invoke({"halloween night at a haunted museum"})

print(image_prompt["text"])

image_url = DallEAPIWrapper(model="dall-e-3").run(image_prompt["text"])
print(image_url)