import logging
import os
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from ..tools.web_loader import search_internet
from ..utils.schemas import (Classifier, Evaluator, PartySentence,
                             QuestionType, SpotifyAction,
                             SpotifyActionClassifier, SpotifyClassifier,
                             Summarizer, Translator)

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
    result, info = model.transcribe(BytesIO(input), beam_size=5, language="ca")
    return result, info


def execute_classifier_agent(
    english_command: str, preestablished_commands_schema: dict, feedback: str
) -> Classifier:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Classifier)

    system = f"""
    You are an assistant and expert text classifier. Classifiy the given command into one of the possible categories.
    - The possible question types are the following:
        {[q.value for q in QuestionType]}
    ## Categories:
    - Spotify Command: Command is related to playing music, pausing music , playing the radio or turining up or down the volume.
    - Knowledge Question: Commands asking about information. It does not include weather related questions.
    - Home Assistantg: Commands asking about a homeassistant or a domotic related task. Related to turning on or off lights or the fan.
    - API Call: Commands asking about the weather, if it will be sunny, rainy, cloudy. Route here if it is asking about hanging the clothes.
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

    system = f"""
    You are an expert evaluator.
    
    ## Task:
    - Think step by step 
    - Assess if the original command and the question type are the most similar between the possible options.
    - If they do not match, give a detailed feedback for improvement.
    - The possible question types are the following. :
        {[q.value for q in QuestionType]}

    ## Categories explanation:
    - Spotify Command: Command is related to playing music, adding music to queue, pausing music , playing the radio or turining up or down the volume, etc.
    - Knowledge Question: Commands asking about information. It does not include weather related questions.
    - Home Assistantg: Commands asking about a homeassistant or a domotic related task. Related to turning on or off lights or the fan.
    - API Call: Commands asking about the weather, if it will be sunny, rainy, cloudy. Route here if it is asking about hanging the clothes.
    - Party Mode: Commands in the list of preestablished commands. These are easter eggs.

    ## Output format:
    - feedback: a short feedback comment to improve the question routing if it is incorrect.
    - is_correct: a boolean stating if the question type decided given is correct.

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

    result: Evaluator = evaluator.invoke(
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
) -> str:
    llm = get_model()

    system = f"""
    You are an expert translator, you will be given a sentence. Translate it from {original_language} to {destination_language}
    
    **Tasks: 
    - Think step by step. 
    - Translate word by word the given text being aware of double meanings in words.
    - Do not translate people's artists' or albums names.
    
    ## Output format
    - "The translated text"

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
    traslator_llm: RunnableSequence = answer_prompt | llm

    result: dict = traslator_llm.invoke(
        {
            "text": text,
        }
    )
    return result.content


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
    You are a task router and classifier.

    ##Task:
    - Decide which action suits best: PLAY (search and play given music) or OTHER (actions not related to a song or group, more like playback related like :pause, next track, previous track, volume...)
    - Return the most similar option

    ## Output Format:
    - classification: either PLAY or OTHER
    - reasoning : str with a short description.
    """
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """{text} """,
            ),
        ]
    )

    classifier: RunnableSequence = answer_prompt | structured_llm_grader

    result: SpotifyActionClassifier = classifier.invoke(
        {
            "text": text,
        }
    )
    logger.info(f"reasoning: {result.reasoning}")
    return result.classification


def execute_tool_agent(
    english_command: str,
    tools: list,
) -> str:
    # tool_description = (
    #     f"{x['name']}:{x['description']}. Args: {x['args']}\n" for x in tools
    # )
    system_prompt = """
    You are a tool calling agent.
    ## Task: 
    - Given a command, decide which tool should be called.
    - Call the tool and provide a short summary sentence of the result.
    """
    agent = create_agent(
        model=get_model(),
        tools=tools,
        system_prompt=system_prompt,
    )
    logging.info("Agent created")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": english_command}]}
    )
    logging.info(response)
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
    - Call the tool and provide a short summary of the result.
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
