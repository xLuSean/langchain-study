# This is a long document we can split up.


from langchain_text_splitters import RecursiveCharacterTextSplitter
# >>> basic usage
# with open("../../text_files/state_of_the_union.txt") as f: state_of_the_union = f.read()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# texts = text_splitter.create_documents([state_of_the_union])

# # print(texts[0])
# # print(texts[1])

# >>> Splitting text from languages without word boundaries
# Some writing systems do not have word boundaries, for example Chinese, Japanese, and Thai. Splitting text with the default separator list of ["\n\n", "\n", " ", ""] can cause words to be split between chunks. To keep words together, you can override the list of separators to include additional punctuation:

# Add ASCII full-stop “.”, Unicode fullwidth full stop “．” (used in Chinese text), and ideographic full stop “。” (used in Japanese and Chinese)
# Add Zero-width space used in Thai, Myanmar, Kmer, and Japanese.
# Add ASCII comma “,”, Unicode fullwidth comma “，”, and Unicode ideographic comma “、”

with open("../../text_files/Neil-Gaiman-I-Cthulhu-CHT.txt") as f: cthulhu = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=60,
    length_function=len,
    separators=[
        "\n\n",
        "\n",
        # " ",
        # ".",
        # ",",
        # "\u200B",  # Zero-width space
        # "\uff0c",  # Fullwidth comma
        # "\u3001",  # Ideographic comma
        # "\uff0e",  # Fullwidth full stop
        # "\u3002",  # Ideographic full stop
        # "。",
    ],
    # Existing args
)

text = text_splitter.create_documents([cthulhu])

for i in range(20):
    print(text[i], "\n")
# print(text[0:5])
# print(len(text))