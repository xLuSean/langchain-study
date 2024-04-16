# This is a long document we can split up.


from langchain_text_splitters import RecursiveCharacterTextSplitter

# with open("../../text_files/Neil-Gaiman-I-Cthulhu-CHT.txt") as f: cthulhu = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=100,
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
        "\u3002",  # Ideographic full stop
        "。",
    ],
    # Existing args
)

input = """當狗狗的屁股散發出一股難以形容的異味時，這往往是來自於肛門腺的分泌物。肛門腺，也稱肛門囊，是位於狗狗肛門四點與八點鐘方向的腺體，主要有三個功能：協助排便、辨別身份和標記地盤。正常情況下，肛門腺液會隨著狗狗的便便自然排出，但有時候可能需要人工協助擠壓。

如果發現狗狗屁屁有腥味、磨屁股、舔或咬屁股、坐立不安或排便困難等症狀，這可能表示需要擠壓肛門腺。擠壓肛門腺時，應輕輕按壓肛門口附近的肛門腺，從裡向外推壓，形成一個ㄈ字形的路徑。如果無法擠出或狗狗表現出不適，應尋求專業獸醫或美容師的協助。

肛門腺如果出現紅腫或發炎，建議立即帶狗狗看獸醫，以避免更嚴重的健康問題。通过這些步驟，可以幫助你的狗狗保持健康，遠離不適。

當狗狗總是喜歡吃棉被或貓糞時，這可能是一種稱為Pica的異食癖症狀，意指亂吃非食物物品。異食癖可能與心理因素有關，如無聊或尋求關注。特別是，狗狗吃自己的糞便是一種常見的現象，可能與其祖先狼的行為有關，狼可能會吃掉糞便以清理巢穴並減少寄生蟲感染。

對於狗狗吃飽了仍然吃異物的情況，飼主應避免過度反應以免強化異食癖行為。例如，當狗狗咬垃圾時，激烈的反應可能會讓狗狗將吃垃圾與獲得關注聯結起來。此外，如果狗狗不慎吞下異物，這可能導致腸道阻塞或傷害，需立即尋求動物醫院的專業治療。

若異食癖情況嚴重，可能需要專業醫療介入，尤其是當異物導致消化道損傷或卡住時。飼主在日常生活中應確保危險物品遠離狗狗的觸及範圍，並提供適合的玩具及足夠的陪伴，以減少狗狗的異食行為。在狗狗亂吃物品時，避免強烈責罵，以免進一步強化不良行為。
"""

texts = text_splitter.split_text(input)

# print(len(texts))

for i in range(len(texts)):
    print(f"chunk {i}:\n {texts[i]}", "\n")
# print(text[0:5])
# print(len(text))