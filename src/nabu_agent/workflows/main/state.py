from typing_extensions import TypedDict

from ...utils.schemas import QuestionType, SpotifyType


class MainGraphState(TypedDict):
    stt_output: str
    input: bytes
    original_language: str
    english_command: str
    routing_ok: bool
    feedback: str
    question_type: QuestionType
    spotify_command: SpotifyType
    spotify_query: str
    web_search: str
    final_answer: str  # sentence to return
    final_answer_translated: str
