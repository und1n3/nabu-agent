import logging

from dotenv import load_dotenv

from ...tools.agents import (execute_spotify_classifier_agent,
                             execute_spotify_decide_action, execute_tool_agent)
from ...tools.spotify import (init_spotify, next_song, pause_music, play_music,
                              previous_song, search_music, volume_down,
                              volume_up)
from ...utils.schemas import SpotifyAction, SpotifyClassifier, SpotifyType
from ...workflows.main.state import MainGraphState

load_dotenv()

logger = logging.getLogger(__name__)


def decide_action(state: MainGraphState) -> MainGraphState:
    logger.info(" --- Decide Spotify action ---")
    result: SpotifyAction = execute_spotify_decide_action(text=state["english_command"])
    logger.info(f"Decided Spotify Action: {result}")
    state["spotify_action"] = result
    return state


def other_functionalities(state: MainGraphState) -> MainGraphState:
    logger.info("--- Other Spotify Commands ---")
    result: str = execute_tool_agent(
        english_command=state["english_command"],
        tools=[pause_music, next_song, previous_song, volume_down, volume_up],
    )
    state["final_answer"] = result
    return state


def decide_music_type(state: MainGraphState) -> MainGraphState:
    logger.info("--- Decide Spotify Command Type Node ---")
    result: SpotifyClassifier = execute_spotify_classifier_agent(
        text=state["english_command"],
    )
    state["spotify_command"] = result.classification
    state["spotify_query"] = result.key_word.replace(" ", "%20")

    logger.info(f"Enrouting to: {result.classification}")
    logger.info(f"Query: {result.key_word}")
    return state


def search_and_play_music(state: MainGraphState) -> MainGraphState:
    spotify_client = init_spotify()
    logger.info("--- Search & Play Song Node ---")
    id = search_music(
        spotify_client,
        query=state["spotify_query"],
        criteria_type=state["spotify_command"],
    )
    if state["spotify_command"] == SpotifyType.TRACK:
        uris = id
        context_uri = None
    else:
        uris = None
        context_uri = id

    play_music(spotify_client, context_uri=context_uri, uris=uris)
    state["final_answer"] = f"Playing Music: {state['english_command']}"
    return state
