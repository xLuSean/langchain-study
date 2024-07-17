# Ref :https://medium.com/@bpothier/generating-structured-data-from-an-image-with-gpt-vision-and-langchain-34aaf3dcb215

import base64
from langchain.chains.transform import TransformChain

# >>> Loading and Encoding the Image
def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
  
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    image_base64 = encode_image(image_path)
    return {"image": image_base64}

load_image_chain = TransformChain(
    input_variables=["image_path"],
    output_variables=["image"],
    transform=load_image
)

# >>> Defining the Output Structure
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class ImageInformation(BaseModel):
    """Information about an image."""
    image_description: str = Field(description="a short description of the image")
    people_count: int = Field(description="number of humans on the picture")
    main_objects: list[str] = Field(description="list of the main objects on the picture")

parser = JsonOutputParser(pydantic_object=ImageInformation)

# >>> Setting up the Image Model
# from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain

# Set verbose
# globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4-vision-preview", max_tokens=1024)
    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                # {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}}
                ]),
                SystemMessage(content = [parser.get_format_instructions()]),
                # SystemMessage(content = ["hello"])  # for testing
                ],
                )
    # print(f"!!!!!!!!\n {parser.get_format_instructions()} \n!!!!!!!!!!")
    return msg.content

# >>> get info chain
def get_image_informations(image_path: str) -> dict:
    vision_prompt = """
    Given the image, provide the following information:
    - A count of how many people are in the image
    - A list of the main objects present in the image
    - A description of the image
    """
    vision_chain = load_image_chain | image_model | parser
    return vision_chain.invoke({'image_path': f'{image_path}', 'prompt': vision_prompt})


# >>> get the result
result = get_image_informations("../example_img/business-people.jpg")
print(result['image_description'])