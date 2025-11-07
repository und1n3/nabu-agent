from src.nabu_agent.tools.agents import execute_knowdledge_agent
from dotenv import load_dotenv

load_dotenv()
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_1():
    english_command = "why is the sky blue?"

    result = await execute_knowdledge_agent(english_command)
    final_answ = result
    print(final_answ)
    assert "sky" in final_answ


@pytest.mark.asyncio
async def test_2():
    english_command = "Who is the current USA president?"

    result = await execute_knowdledge_agent(english_command)
    final_answ = result
    print(final_answ)
    assert "Trump" in final_answ


@pytest.mark.asyncio
async def test_3():
    english_command = "Who are the current top 3 spotify artists?"

    result = await execute_knowdledge_agent(english_command)
    final_answ = result
    print(final_answ)
    assert "Swift" in final_answ
