from typing import Annotated,TypedDict, Sequence
from langchain_core.messages import BaseMessage,HumanMessage,SystemMessage
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition

from tools import pdf_retrieval_tool

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def get_llm(provider:str,api_key:str):
    """ Instantiate LLM based on the chosen LLm Provider."""
    if provider == "Google_Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model = "gemini-3.6-flash",
            google_api_key = api_key,
            temperature = 0.2
        )

    else:
        raise ValueError(f"Unsupported provider selected")

def create_multimodal_rag_graph(provider:str,api_key:str) :
    " Builds and  compile the langgraph state graph"
    tools = [pdf_retrieval_tool]
    llm = get_llm(provider,api_key)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state:AgentState):
        system_prompt = SystemMessage(content=(
            "You are a expert multimodel documnet assistant capable pf analyzing text"
            "tables,graphs and images inside pdfs."
            "Always use the pdf_retrieval_tool to retrieve relevant information from the uploaded pdfs."
            "to uploaded files.If retrived results conatin image base64 data ,analyze visual content carefully."  
        ))

        messages = [system_prompt] + list(state["messages"])
        response = llm_with_tools.invoke(messages)

        if provider == "Google_Gemini":
            return {"messages": [response]}

        else:
            return {"messages": [response.content[0]["text"]]}

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))


    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)

    return workflow.compile()
