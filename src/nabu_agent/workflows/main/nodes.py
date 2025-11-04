import logging

from dotenv import load_dotenv

from ...data.preestablished_commands import party_commands
from ...tools.agents import (
    execute_classifier_agent,
    execute_ha_command,
    execute_party_sentence,
    execute_search_text,
    execute_stt,
    execute_translator,
)
from ...utils.schemas import Classifier, PartySentence, QuestionType, Translator
from ...workflows.main.state import MainGraphState

load_dotenv()

logger = logging.getLogger(__name__)


def stt(state: MainGraphState) -> MainGraphState:
    logger.info("--- Whisper Speech To Text --- ")
    result = execute_stt(input=state["input"])
    final_result = ""
    for i in result:
        logger.info(i.text)
        final_result += i.text
    state["stt_output"] = final_result
    return state


def translate_to_english(state: MainGraphState) -> MainGraphState:
    logger.info("--- Translating to english --- ")
    result: Translator = execute_translator(
        text=state["stt_output"], destination_language="english"
    )
    state["original_language"] = result.original_language
    state["english_command"] = result.translated_command
    logger.info(
        f"Language detected: {result.original_language}. Translated text: {result.translated_command}"
    )
    return state


def enroute_question(state: MainGraphState) -> MainGraphState:
    logger.info("--- Enroute Question Node ---")
    result: Classifier = execute_classifier_agent(
        english_command=state["english_command"],
        preestablished_commands_schema=party_commands,
    )
    state["question_type"]: QuestionType = result.classification
    logger.info(f"Enrouting to: {result.classification}")
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


def internet_search(state: MainGraphState) -> MainGraphState:
    logger.info("--- Internet Search Node ---")

    search: str = execute_search_text(english_command=state["english_command"])
    logger.info(f"Result from the web search: {search}")

    state["final_answer"] = search

    return state


def finish_action(state: MainGraphState) -> MainGraphState:
    logger.info("--- Final Action Node ---")
    logger.info("Translating the final answer to reproduce in the speakers.")
    if "final_answer" not in state:
        state["final_answer"] = state["english_command"]
    logger.info(f"Sentence: {state['final_answer']}")

    result = execute_translator(
        text=state["final_answer"],
        destination_language=state["original_language"],
    )
    state["final_answer_translated"] = result.translated_command
    logger.info(f"Translated Sentence: {state['final_answer_translated']}")

    return state


async def homeassistant(state: MainGraphState) -> MainGraphState:
    logger.info("--- Home Assistant Node ---")

    search: str = await execute_ha_command(english_command=state["english_command"])
    logger.info(f"Result from HA : {search}")

    state["final_answer"] = search

    return state
