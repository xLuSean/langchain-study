from langchain.prompts import PromptTemplate

# method 1 ==============================================================================

# template = """{test}
# 我希望你能充当新公司的命名顾问。
# 一个生产{product}的公司的好名字是什么？
# """

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template=template,
# )
# res = prompt.format(product="彩色袜子", test="你好")
# print(res)

# method 2 ==============================================================================
# 如果我们不想手动指定input_variables，还可以使用from_template类方法创建PromptTemplate。LangChain将根据传递的template自动推断input_variables。

template = "给我讲一个{adjective}的关于{content}的笑话。"
prompt_template = PromptTemplate.from_template(template)
test = prompt_template.input_variables
print(test)
# -> ['adjective', 'content']
prompt_template.format(adjective="有趣的", content="小鸡")
# -> 给我讲一个有趣的关于小鸡的笑话。

# prompt_template.save("prompt_template.json")

from langchain.prompts import load_prompt
prompt = load_prompt("./prompt_template.json")
print(prompt.format(adjective="有趣的", content="小鸡"))