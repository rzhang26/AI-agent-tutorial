from dotenv import load_dotenv 
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser 
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
parser = PydanticOutputParser(pydantic_object=BaseModel)

response = llm.invoke()
print(response)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a reseach assistant that will help compile succint and relevant information.
            Answer the user query and use nesisary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}

            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instruction = parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(
    LLm = llm,
    prompt = prompt,
    tools = tools
)

agent_executer = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What may I help you research?")
raw_response = agent_executer.invoke({"query": query})

try:
    structured_response = parser.parse(raw_response.get("output")[0]("text"))
    print(structured_response)
except Exception as e:
    print("error parsing response: ", e, "raw response: ", raw_response)
