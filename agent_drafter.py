from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_community.chat_models.tongyi import ChatTongyi

document_content = " "

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
        """Update the document with the provided content."""
        global document_content
        document_content = content
        return f"Document has been updated successfully! The current content is: {document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """
    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename,'w') as file:
              file.write(document_content)
        print(f"\n Document has been saved to {filename} successfully!")
        return f"Document has been saved to {filename} successfully!"
    except Exception as e:
        return f"Error saving document: {str(e)}"
    

tools = [update, save]
model = ChatTongyi(model="qwen-plus").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are a helpful writing assistant. You are going to help the user update and modify a document.
    
    - If the user wants to update or modify content, use the `update` tool with the complete content.
    - If the user wants to save the document, use the `save` tool with the desired filename.
    - Make sure to always show the current document state after modification.
        
    The current document content is: {document_content}
                                  """)
    if not state["messages"]:
         user_input = "I'm ready to help you with your document. Please provide the content you want to update or modify."
         user_message = HumanMessage(content=user_input)
    else:
         user_input = input("What would you like to do with the document? ")
         print(f"\n User input: {user_input}")
         user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n Model AI: {response.content}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"\n Using Tool : {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end the conversation."""
    messages = state["messages"]

    if not messages:
         return "continue"
    for message in reversed(messages):
         if (isinstance(message, ToolMessage) and
             "saved" in message.content.lower() and
             "document" in message.content.lower()):
             return "end"
         
    return "continue"

def print_messages(messages):
     """Function I made to print messages in a readable format."""
     if not messages:
          return
     
     for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n Tool Result: {message.content}")

graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ================== Document Agent =================")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
        
    print("\n ================== Document Agent Finished =================")

if __name__ == "__main__":
    run_document_agent()
    