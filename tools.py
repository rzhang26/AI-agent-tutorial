from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool 

from datetime import datetime

# Search tool browser
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name = "search",
    func = search.run,
    description = "search web for detailed info"
)

# Search tool wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


# Save tool function
def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    formatted_text = f"--- Research Output ---\n Timestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding = "utf-8") as f:
        f.write(formatted_text)

    return f"data successfully saved to {filename}"

# Save tool
save_tool = Tool(
    name = "save-text-to-file",
    func = save_to_txt,
    description = "save structured rsh data to .txt file"
)