from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

# search = DuckDuckGoSearchResults()
search = DuckDuckGoSearchResults(backend="news")

res = search.run("Obama's first name?")

print(res)