from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate

from chat.customer_support.tools import (fetch_user_flight_information, search_flights, lookup_policy, update_ticket_to_new_flight,
    cancel_ticket, search_car_rentals, book_car_rental, update_car_rental, cancel_car_rental, search_hotels, book_hotel, 
    update_hotel, cancel_hotel, search_trip_recommendations, book_excursion, update_excursion, cancel_excursion,)
from chat.llm import get_llm, LLMInferenceProvider
from chat.customer_support.agents import State, StateV2, ZeroShotAssistant, AssistantV2
from chat.customer_support.utils import create_tool_node_with_fallback
from chat.customer_support.tools import fetch_user_flight_information

llm = get_llm(LLMInferenceProvider.AZURE_OPENAI)
# db = 

zero_shot_tools = [
    # TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    update_ticket_to_new_flight,
    cancel_ticket,
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel,
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion,
] 



primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

zero_shot_assistant_runnable = primary_assistant_prompt | llm.bind_tools(zero_shot_tools)

def get_zero_shot_graph():
    """Get the zero-shot graph."""
    builder = StateGraph(State)

    # define the nodes, these do the work
    builder.add_node("assistant", ZeroShotAssistant(zero_shot_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(zero_shot_tools))

    # define edges: These define how the control flow moves
    builder.set_entry_point("assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    # this checkpointer lets the graph persis its state. 
    # This is a complete memory for the entire graph
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = builder.compile(checkpointer=memory)
    return graph

def get_user_info(state: State):
    return {"user_info": fetch_user_flight_information.invoke({})}

def get_graph_with_confirmation_v2():
    '''same as zero shot graph, but with user confirmation'''
    builder = StateGraph(StateV2)

    builder.add_node("fetch_user_info", get_user_info)
    builder.set_entry_point("fetch_user_info")
    builder.add_node("assistant", AssistantV2(zero_shot_assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(zero_shot_tools))
    builder.add_edge("fetch_user_info", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    memory = SqliteSaver.from_conn_string(":memory:")
    graph_v2 = builder.compile(checkpointer=memory, interrupt_before=["tools"]) 
    return graph_v2





