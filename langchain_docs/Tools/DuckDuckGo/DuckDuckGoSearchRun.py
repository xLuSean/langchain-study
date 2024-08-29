from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# res = search.run("Obama's first name?")
res = search.run("貓飼料排行榜")

print(res)

# To get more additional information (e.g. link, source) use DuckDuckGoSearchResults()