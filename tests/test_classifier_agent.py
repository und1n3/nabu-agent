import json

from src.nabu_agent.tools.agents import execute_classifier_agent
from src.nabu_agent.utils.schemas import Classifier, QuestionType

preestablished_commands_schema = json.load(open("data/preestablished_commands.json"))


def test_classifier_agent_internet_1():
    input_prompt = "Quin temps fa avui a mataró?"
    result: Classifier = execute_classifier_agent(
        text=input_prompt, preestablished_commands_schema=preestablished_commands_schema
    )
    assert result.classification == QuestionType.api_call


def test_classifier_agent_internet_2():
    input_prompt = "quin any va ser la revolució francesa?"
    result: Classifier = execute_classifier_agent(
        text=input_prompt, preestablished_commands_schema=preestablished_commands_schema
    )
    assert result.classification == QuestionType.knowledge


def test_classifer_agent_spotify_1():
    input_prompt = "Posa una cançó de Mika"
    result: Classifier = execute_classifier_agent(
        text=input_prompt, preestablished_commands_schema=preestablished_commands_schema
    )
    assert result.classification == QuestionType.spotify


def test_classifer_agent_party_1():
    input_prompt = "boom"
    result: Classifier = execute_classifier_agent(
        text=input_prompt, preestablished_commands_schema=preestablished_commands_schema
    )
    assert result.classification == QuestionType.party
