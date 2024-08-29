from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults

# wrapper = DuckDuckGoSearchAPIWrapper(region="de-de", time="d", max_results=2)
wrapper = DuckDuckGoSearchAPIWrapper(max_results=2)

search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

# res = search.run("Obama")
res = search.run("推薦貓飼料")

print(res)