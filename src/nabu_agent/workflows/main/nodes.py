import logging

from dotenv import load_dotenv

from ...data.preestablished_commands import party_commands
from ...tools.agents import (
    execute_classifier_agent,
    execute_evaluator_agent,
    execute_ha_command,
    execute_knowdledge_agent,
    execute_party_sentence,
    execute_stt,
    execute_tool_agent,
    execute_translator,
)
from ...tools.misc import get_weather
from ...utils.schemas import (
    Classifier,
    Evaluator,
    PartySentence,
    QuestionType,
    Translator,
)
from ...workflows.main.state import MainGraphState

load_dotenv()

logger = logging.getLogger(__name__)


def stt(state: MainGraphState) -> MainGraphState:
    logger.info("--- Whisper Speech To Text --- ")
    result, info = execute_stt(input=state["input"])
    state["input"] = None
    final_result = ""
    for i in result:
        final_result += i.text
    state["stt_output"] = final_result
    state["original_language"] = "spanish" if info.language == "es" else "catalan"
    logger.info(f"Transcription from {info.language}: {final_result}")
    return state


def translate_to_english(state: MainGraphState) -> MainGraphState:
    logger.info("--- Translating to english --- ")
    result: str = execute_translator(
        text=state["stt_output"],
        destination_language="english",
        original_language=state["original_language"],
    )
    state["english_command"] = result
    logger.info(
        f"Translated text (from {state['original_language']} to english): {result}"
    )
    return state


def enroute_question(state: MainGraphState) -> MainGraphState:
    logger.info("--- Enroute Question Node ---")
    result: Classifier = execute_classifier_agent(
        english_command=state["english_command"],
        preestablished_commands_schema=party_commands,
        feedback=state.get("feedback", None),
    )
    state["question_type"] = result.classification
    logger.info(f"Category Classification: {result.classification}")
    return state


def verify_routing(state: MainGraphState) -> MainGraphState:
    logger.info("--- Evaluating Routing ---")
    state["retries"] = state.get("retries", 0)
    state["retries"] += 1
    result: Evaluator = execute_evaluator_agent(
        original_command=state["english_command"],
        question_type=state.get("question_type", None),
    )
    state["routing_ok"] = result.is_correct
    state["feedback"] = result.feedback
    return state


def pre_established_commands(state: MainGraphState) -> MainGraphState:
    logger.info("--- Pre-Established Commands Node ---")
    result: PartySentence = execute_party_sentence(
        text=state["english_command"],
        preestablished_commands_schema=party_commands,
    )
    logger.info(f"Command Used: {result.command_used}")
    state["final_answer"] = result.sentence
    return state


async def knowledge_answerer(state: MainGraphState) -> MainGraphState:
    logger.info("--- Knowdledge Node ---")

    search: str = await execute_knowdledge_agent(
        english_command=state["english_command"]
    )
    logger.info(f"Result from the web search: {search}")

    state["final_answer"] = search

    return state


def api_call(state: MainGraphState) -> MainGraphState:
    """Tool Calling agent. External APIs"""
    tools = [get_weather]
    result = execute_tool_agent(state["english_command"], tools)
    state["final_answer"] = result
    return state


def finish_action(state: MainGraphState) -> MainGraphState:
    logger.info("--- Final Action Node ---")
    logger.info("Translating the final answer to reproduce in the speakers.")
    if "final_answer" not in state:
        state["final_answer"] = state["english_command"]
    logger.info(f"Sentence: {state['final_answer']}")

    result: str = execute_translator(
        text=state["final_answer"],
        destination_language=state[
            "original_language"
        ],  # original language of the user voice command
        original_language="english",
    )
    state["final_answer_translated"] = result
    logger.info(f"Translated Sentence: {state['final_answer_translated']}")

    return state


async def homeassistant(state: MainGraphState) -> MainGraphState:
    logger.info("--- Home Assistant Node ---")

    search: str = await execute_ha_command(english_command=state["english_command"])
    logger.info(f"Result from HA : {search}")

    state["final_answer"] = search

    return state
