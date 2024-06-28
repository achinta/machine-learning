from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.prompts import ChatPromptTemplate

from chat.customer_support.tools import *
from chat.llm import get_llm, LLMInferenceProvider
from chat.customer_support.agents import State, ZeroShotAssistent
from chat.customer_support.utils import create_tool_node_with_fallback

llm = get_llm(LLMInferenceProvider.AZURE_OPENAI)

zero_shot_tools = [
    # TavilySearchResults(max_results=1),÷
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

zero_shot_assistnat_runnable = primary_assistant_prompt | llm.bind_tools(zero_shot_tools)

def get_zero_shot_graph():
    """Get the zero-shot graph."""
    builder = StateGraph(State)
    
    # define the nodes
    builder.add_node("assistant", ZeroShotAssistent(zero_shot_assistnat_runnable))
    builder.add_node("tools", create_tool_node_with_fallback)

    # define the edges
    builder.set_entry_point("assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # The checkpointer lets the graph persist its state
    # this is a complete memory for the entire graph.
    memory = SqliteSaver.from_conn_string(":memory:")
    graph = builder.compile(checkpointer=memory)
    return graph

    

