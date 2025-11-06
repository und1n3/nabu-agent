from src.nabu_agent.tools.agents import execute_ha_command
from dotenv import load_dotenv

load_dotenv()
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_1():
    english_command = "Is the Potus watered? (it has a sensor)"

    result = await execute_ha_command(english_command)
    final_answ = result
    print(final_answ)
    assert "weather" in final_answ


@pytest.mark.asyncio
async def test_2():
    english_command = "List all devices available at home."

    result = await execute_ha_command(english_command)
    final_answ = result
    print(final_answ)
    assert "weather" in final_answ
