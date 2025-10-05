import logging
import os
from io import BytesIO

from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from ..tools.web_loader import search_and_fetch
from ..utils.schemas import (
    Classifier,
    PartySentence,
    QuestionType,
    SpotifyClassifier,
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
        top_p=1,
    )
    return model


def execute_stt(input: bytes):
    # Run on GPU with FP16
    if os.getenv("FASTER_WHISPER_USE_CUDA") == "true":
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    else:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    result, info = model.transcribe(BytesIO(input), beam_size=5)
    return result


def execute_classifier_agent(
    english_command, preestablished_commands_schema
) -> Classifier:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Classifier)

    system = """
    You must assess if the given text is one of the prestablished commands, a question that needs internet access or a command to play music in spotify.
    The preestablished commands are in the format {{trigger_sentence : description of the type of answer you have to say}}
            
            - preestablished_command  (exact match or fuzzy match; return matched_trigger)
            - spotify                (user wants to play music/radio)
            - web_search             (user asks a general/up-to-date question requiring internet)

    **INSTRUCTIONS**:
    - Think thoroughly what time of command you are asked. it should fall within one of the given categories
    - Decide whether it should go for the spotify command (playing music, play radio), for a search result in the web (a question about general or up to date knowledge), if it is found within the prestablished commands (given list)
      or is a homeassitant_command (turn on the light, what devices I have?)
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """- command:  {text}
                  - prestablished commands: {preestablished_commands_schema}
                  - Question categories: {question_categories}
                  """,
            ),
        ]
    )

    classifier: RunnableSequence = answer_prompt | structured_llm_grader

    result: Classifier = classifier.invoke(
        {
            "preestablished_commands_schema": preestablished_commands_schema,
            "text": english_command,
            "question_categories": QuestionType,
        }
    )
    return result


def execute_search_text(english_command):
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Summarizer)
    search_result = search_and_fetch(
        query=english_command, num_results=2, chunk_size=1500
    )

    logger.info(f"\n\n Search Result: {search_result}")
    # Define the prompt template for the agent
    system = """
    You are an expert in summarizing content and giving the most accurate information. Given an initial command an the text containing the information,
    Give an answer to the command using the information provided in the text. Keep it short, around one sentence.
    
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """ Command: {command}
                    Text: \n\n {text} """,
            ),
        ]
    )

    agent = answer_prompt | structured_llm_grader
    result = agent.invoke({"command": english_command, "text": search_result})
    logger.info(f"Answer: {result.answer}")
    return result.answer


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


def execute_translator(text: str, destination_language: str) -> Translator:
    llm = get_model()
    structured_llm_grader = llm.with_structured_output(Translator)

    system = """
    You are a language expert, you will be working mainly in catalan and english.Think thoroughly the meaning of the sentence and translate it to the best of your abilities. 
    Translate word by word the given text. Be aware of double meanings in words, choose the correct translation.
    Perform an intelligent translation, some words may be in another language if they are a song or arstist. take this into account.
    
    **INSTRUCTIONS:**
    - Detect the language of the given sentence 
    - translate the text from the original language to the destination language. Return just the text translated.
    - Do not translate people's artists' or albums names.
    
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """
                - Text to translate: {text}
                - Destination language: {destination_language}
                """,
            ),
        ]
    )
    traslator_llm: RunnableSequence = answer_prompt | structured_llm_grader

    result: Translator = traslator_llm.invoke(
        {
            "text": text,
            "destination_language": destination_language,
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
    Instructions:
    - Translate the question into english.
    - Decide which of the given Spotify commands suits best [radio, song, playlist or album]. If radio mentioned, type is radio.
    - Return also the key word(s) to search for. For example, given a song it would be the song title, radio or playlist would be the name, album would be title . Also the artist if given
    
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


async def execute_ha_command(english_command):
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

    llm = get_model().bind_tools(tools)
    result = await llm.ainvoke(english_command)
    for tool_call in result.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        tool = next(t for t in tools if t.name == tool_name)
        try:
            tool_result = await tool.ainvoke(tool_args)

            followup = await llm.ainvoke(
                [
                    f"The original prompt is: {english_command}, parse the result of the tool following the instructions:",
                    result,
                    tool_result,
                ]
            )
        except Exception:
            return "Error calling tools in HA command"
        return followup.content
    return result
