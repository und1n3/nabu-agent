import logging
import os
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from ..tools.web_loader import search_internet
from ..utils.schemas import (
    Classifier,
    Evaluator,
    PartySentence,
    QuestionType,
    SpotifyClassifier,
    SpotifyAction,
    SpotifyActionClassifier,
    Summarizer,
    Translator,
)

logger = logging.getLogger(__name__)
model_size = os.getenv("FASTER_WHISPER_MODEL")
load_dotenv()


def get_model() -> ChatOpenAI:
    # model = ChatOllama(
    #     model="qwen3:30b", temperature=0.15, top_p=1 - 0.01, num_ctx=8192
    # )
    # model = ChatOllama(model="llama3.2")
    model = ChatOpenAI(
        # model="GPT-OSS-20B",
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=0.1,
        top_p=0.5,
    )
    return model


def execute_stt(input: bytes):
    # Run on GPU with FP16
    if os.getenv("FASTER_WHISPER_USE_CUDA") == "true":
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    else:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    result, info = model.transcribe(BytesIO(input), beam_size=5)
    return result, info


def execute_classifier_agent(
    english_command: str, preestablished_commands_schema: dict, feedback: str
) -> Classifier:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Classifier)

    system = """
    You are an assistant and expert text classifier. Classifiy the given command into one of the possible categories.
    ## Categories:
    - Spotify Command: Command is related to playing music, pausing music , playing the radio or turining up or down the volume.
    - Knowledge Question: Commands asking about information. It does not include weather related questions.
    - Domotics Routing: Commands asking about a homeassistant or a domotic related task. Related to turning on or off lights or the fan.
    - Weather API: Commands asking about the weather, if it will be sunny, rainy, cloudy. Route here if it is asking about hanging the clothes.
    - Party Mode: Commands in the list of preestablished commands. These are easter eggs.

    ## Task
    - Think step by step.
    - Route the input command into the most appropiate category type.
    - You may be given feedback to help with the routing, use it to decide the category.

    ## Output Format
    - classification : Question type category.
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """
                  - Command:  {english_command}
                  - Prestablished commands: {preestablished_commands_schema}
                  - feedback: {feedback}
                  """,
            ),
        ]
    )

    classifier: RunnableSequence = answer_prompt | structured_llm_grader

    result: Classifier = classifier.invoke(
        {
            "preestablished_commands_schema": preestablished_commands_schema,
            "english_command": english_command,
            "feedback": feedback,
        }
    )
    return result


def execute_evaluator_agent(
    original_command: str, question_type: QuestionType
) -> Evaluator:
    llm = get_model()
    structured_llm_evaluator = llm.with_structured_output(Evaluator)

    system = """
    You are an expert evaluator.
    
    ## Task:
    - Assess if the original command and the question type match.
    - If they do not match, give a detailed feedback for improvement.

    ## Output format:
    - is_correct: a boolean stating if the question type decided given is correct.
    - feedback: a feedback comment to improve the question routing.
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """
                  - Original command: {original_command}
                  - Routed Question Type: {question_type}
                  """,
            ),
        ]
    )

    evaluator: RunnableSequence = answer_prompt | structured_llm_evaluator

    result: Classifier = evaluator.invoke(
        {
            "original_command": original_command,
            "question_type": question_type.value,
        }
    )
    return result


async def execute_knowdledge_agent(english_command):
    system_prompt = f"""
    You are a knowledgeable and reliable expert assistant with access to an internet search tool for retrieving up-to-date information. 
    Currently we are at {datetime.today()}, if the knowledge for the question is time dependant, use the tool.

    ## Task:
    - If the question can be answered from your world knowledge and independently of the current date, respond directly.
    - Otherwise, if the question requires **recent, specific, or factual data** (e.g., about events, prices, statistics, or companies), use the `search_internet` tool to gather accurate information.
    -  When possible, **mention sources or inferred confidence** (e.g., “According to recent reports...” or “Multiple sources agree...”)
    - Provide a final answer with the available information

    ## Output format
    - If the tool was used, say you searched the internet along with the final answer.
    """
    agent = create_agent(
        model=get_model(),
        tools=[search_internet],
        system_prompt=system_prompt,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": english_command}]}
    )

    return result["messages"][-1].content


def execute_party_sentence(text, preestablished_commands_schema) -> PartySentence:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(PartySentence)

    system = """
    You must assess which prestablished command the given text matches the best.
    Then , following the command's description, return a witty answer .
    Think this is a party / joke mode.
    The preestablished commands are in the format {{trigger_sentence : description of the type of answer you have to say}}
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """User question: \n\n {text} \n\n prestablished commands: {preestablished_commands_schema}""",
            ),
        ]
    )

    party_model: RunnableSequence = answer_prompt | structured_llm_grader

    result: PartySentence = party_model.invoke(
        {
            "preestablished_commands_schema": preestablished_commands_schema,
            "text": text,
        }
    )
    return result


def execute_translator(
    text: str, destination_language: str, original_language: str = "english"
) -> Translator:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Translator)

    system = f"""
    You are an expert translator, you will be translating from {original_language} to {destination_language}
    **Tasks: 
    - Think step by step. 
    - Translate word by word the given text being aware of double meanings in words.
    - Translate the text from the original language to the destination language.
    - Do not translate people's artists' or albums names.
    
    ## Output format
    - Translated text : The original sentence translated to {destination_language}

    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """
                - Text to translate: {text}
                """,
            ),
        ]
    )
    traslator_llm: RunnableSequence = answer_prompt | structured_llm_grader

    result: Translator = traslator_llm.invoke(
        {
            "text": text,
        }
    )
    return result


def execute_spotify_classifier_agent(text) -> SpotifyClassifier:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(SpotifyClassifier)

    system = """
    You must prepare a spotify query search from the user command. Some examples would be:
    Example1 : 
        - Command: play music of la oreja de van gogh
        - Answer: {{'type': artist , 'key_word'  : la oreja de van gogh}}
    Example 2:
        - Command: play the song por qué te vas
        - Answer: {{type: track}}
    Example 3:
        - Command: play radio Crim
        - Answer: {{type: radio}}
    
    ##Task:
    - Decide which of the given Spotify types suits best [radio, song, playlist or album]. If radio mentioned, type is radio.
    - Return also the key word(s) to search for. For example, given a song it would be the song title, radio or playlist would be the name, album would be title . Also the artist if given
    
    ## Output Format:
    - classification: One of "track", "album", "artist", "playlist" or "radio"
    - key word: the name of the artist, playlist, track , radio or album.
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """User command: \n\n {text} """,
            ),
        ]
    )

    classifier: RunnableSequence = answer_prompt | structured_llm_grader

    result: SpotifyClassifier = classifier.invoke(
        {
            "text": text,
        }
    )

    return result


def execute_spotify_decide_action(text) -> SpotifyAction:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(SpotifyActionClassifier)

    system = """
    You must decide if the user query is to play music or to do other actions such as pause music, get next track or previous track, turn volume up or down.
    
    ##Task:
    - Decide which action suits best: PLAY (play music) or OTHER (pause, next, volume...)

    ## Output Format:
    - classification: either PLAY or OTHER
    """
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """User command: \n\n {text} """,
            ),
        ]
    )

    classifier: RunnableSequence = answer_prompt | structured_llm_grader

    result: SpotifyActionClassifier = classifier.invoke(
        {
            "text": text,
        }
    )

    return result.classification


def execute_tool_agent(english_command: str, tools: list) -> str:
    # tool_description = (
    #     f"{x['name']}:{x['description']}. Args: {x['args']}\n" for x in tools
    # )
    system_prompt = """
    You are a tool calling agent.
    ## Task: 
    - Given a command, decide which tool should be called.
    - Call the tool and provide the final result.
    """
    agent = create_agent(
        model=get_model(),
        tools=tools,
        system_prompt=system_prompt,
    )
    response = agent.invoke(
        {"messages": [{"role": "user", "content": english_command}]}
    )
    return response["messages"][-1].content


async def execute_ha_command(english_command: str) -> str:
    ha_token = os.environ["HA_TOKEN"]
    ha_url = os.environ["HA_URL"]
    client = MultiServerMCPClient(
        {
            "homeassistant": {
                "url": f"{ha_url}/mcp_server/sse",
                "transport": "sse",
                "headers": {
                    "Authorization": f"Bearer {ha_token}",
                },
            },
        }
    )
    tools = await client.get_tools()
    system_prompt = """
    You are a tool calling agent. You are a given set of tools and should choose the most adient one.

    ## Task: 
    - Given a command, decide which tool should be called.
    - If none matches the command, use the most similar one.
    - Call the tool and provide a summary of the result.
    """
    agent = create_agent(
        model=get_model(),
        tools=tools,
        system_prompt=system_prompt,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": english_command}]}
    )

    return result["messages"][-1].content
