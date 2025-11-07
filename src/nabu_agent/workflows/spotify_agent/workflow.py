from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ...utils.schemas import SpotifyAction
from ...workflows.main.state import MainGraphState
from ...workflows.spotify_agent import nodes as nodes

load_dotenv()


def decide_action(state: MainGraphState) -> SpotifyAction:
    return state["spotify_action"].value


def build_spotify_workflow() -> CompiledStateGraph:
    workflow = StateGraph(MainGraphState)
    workflow.add_node("Decide Action", nodes.decide_action)
    workflow.add_node("Other Actions", nodes.other_functionalities)
    workflow.add_node("What to play?", nodes.decide_music_type)
    workflow.add_node("Search and play", nodes.search_and_play_music)

    workflow.add_conditional_edges(
        "Decide Action",
        decide_action,
        {
            SpotifyAction.OTHER.value: "Other Actions",
            SpotifyAction.PLAY.value: "What to play?",
        },
    )
    workflow.set_entry_point("Decide Action")
    workflow.add_edge("What to play?", "Search and play")
    workflow.add_edge("Search and play", END)
    workflow.add_edge("Other Actions", END)

    return workflow.compile()
