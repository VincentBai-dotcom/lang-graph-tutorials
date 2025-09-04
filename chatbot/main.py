import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


from dotenv import load_dotenv


def load_env():
    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(
            "Enter API key for Google Gemini: "
        )


def main():
    load_env()
    print("Hello from chatbot!")
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "abc123"}}
    query = "Hi! I'm Bob."

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state

    query = "What's my name?"
    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
