from langchain.prompts import PromptTemplate, FewShotPromptTemplate

# 首先，创建Few Shot示例列表
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# 接下来，我们指定用于格式化示例的模板。
# 我们使用`PromptTemplate`类来实现这个目的。
example_formatter_template = """Word: {word}
Antonym: {antonym}
"""

example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# 最后，创建`FewShotPromptTemplate`对象。
few_shot_prompt = FewShotPromptTemplate(
    # 这些是我们要插入到提示中的示例。
    examples=examples,
    # 这是我们在将示例插入到提示中时要使用的格式。
    example_prompt=example_prompt,
    # 前缀是出现在提示中示例之前的一些文本。
    # 通常，这包括一些说明。
    prefix="Give the antonym of every input\n",
    # 后缀是出现在提示中示例之后的一些文本。
    # 通常，这是用户输入的地方。
    suffix="Word: {input}\nAntonym: ",
    # 输入变量是整个提示期望的变量。
    input_variables=["input"],
    # 示例分隔符是我们将前缀、示例和后缀连接在一起的字符串。
    example_separator="\n",
)

# 现在，我们可以使用`format`方法生成一个提示。
print(few_shot_prompt.format(input="big"))
# -> Give the antonym of every input
# -> 
# -> Word: happy
# -> Antonym: sad
# ->
# -> Word: tall
# -> Antonym: short
# ->
# -> Word: big
# -> Antonym:

from langchain.prompts.example_selector import LengthBasedExampleSelector

# 这里是一些虚构任务的大量示例，该任务是创建反义词。
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

# 我们将使用`LengthBasedExampleSelector`来选择示例。
example_selector = LengthBasedExampleSelector(
    # 这些是可供选择的示例。
    examples=examples, 
    # 这是用于格式化示例的PromptTemplate。
    example_prompt=example_prompt, 
    # 这是格式化示例的最大长度。
    max_length=25
    # 这是用于获取字符串长度的函数，用于确定包含哪些示例。在这里被注释掉，因为如果没有指定，默认提供了该函数。
    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
)

# 现在，我们可以使用`example_selector`创建`FewShotPromptTemplate`。
dynamic_prompt = FewShotPromptTemplate(
    # 我们提供一个ExampleSelector而不是示例。
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入的反义词",
    suffix="单词：{input}\n反义词：",
    input_variables=["input"],
    example_separator="\n\n",
)

# 现在，我们可以使用`format`方法生成提示。
print(dynamic_prompt.format(input="big"))
# -> 给出每个输入的反义词
# ->
# -> 单词：happy
# -> 反义词：sad
# ->
# -> 单词：tall
# -> 反义词：short
# ->
# -> 单词：energetic
# -> 反义词：lethargic
# ->
# -> 单词：sunny
# -> 反义词：gloomy
# ->
# -> 单词：windy
# -> 反义词：平静
# ```python
print(dynamic_prompt.format(input="big"))
# -> 给出每个输入的反义词
# ->
# -> 单词：happy
# -> 反义词：sad
# ->
# -> 单词：tall
# -> 反义词：short
# ->
# -> 单词：energetic
# -> 反义词：lethargic
# ->
# -> 单词：sunny
# -> 反义词：gloomy
# ->
# -> 单词：windy
# -> 反义词：calm
# ->
# -> 单词：big
# -> 反义词：平静
# 相反，如果我们提供一个非常长的输入，LengthBasedExampleSelector会选择较少的示例包含在提示中。

long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
print(dynamic_prompt.format(input=long_string))
# -> 给出每个输入的反义词
# ->
# -> 单词：happy
# -> 反义词：sad
# ->
# -> 单词：big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else
# -> 反义词：平静
