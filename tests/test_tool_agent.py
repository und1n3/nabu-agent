from src.nabu_agent.tools.misc import get_weather
from src.nabu_agent.tools.agents import execute_tool_agent


def test_1():
    english_command = "What's the weather in barcelona spain"
    tools = [
        {
            "name": "get_weather()",
            "description": "Call openweather api to retrieve the city's weather.",
            "args": {"city": "City Name", "date": "Literal: 'today' or 'tomorrow'"},
            "func": get_weather,
        }
    ]
    result = execute_tool_agent(english_command, tools)
    final_answ = result["messages"][-1].content
    assert "weather" in final_answ and "Barcelona" in final_answ


def test_2():
    english_command = "What's the weather in mataro"
    tools = [
        {
            "name": "get_weather()",
            "description": "Call openweather api to retrieve the city's weather.",
            "args": {"city": "City Name", "date": "Literal: 'today' or 'tomorrow'"},
            "func": get_weather,
        }
    ]
    result = execute_tool_agent(english_command, tools)
    final_answ = result["messages"][-1].content
    assert "weather" in final_answ and "Mataró" in final_answ


def test_3():
    english_command = "Will it rain today in calella"
    tools = [
        {
            "name": "get_weather()",
            "description": "Call openweather api to retrieve the city's weather.",
            "args": {"city": "City Name", "date": "Literal: 'today' or 'tomorrow'"},
            "func": get_weather,
        }
    ]
    result = execute_tool_agent(english_command, tools)
    final_answ = result["messages"][-1].content
    assert "weather" in final_answ and "Calella" in final_answ


def test_4():
    english_command = "What's the weather today"
    tools = [
        {
            "name": "get_weather()",
            "description": "Call openweather api to retrieve the city's weather.",
            "args": {"city": "City Name", "date": "Literal: 'today' or 'tomorrow'"},
            "func": get_weather,
        }
    ]
    result = execute_tool_agent(english_command, tools)
    final_answ = result["messages"][-1].content
    assert "weather" in final_answ and "Mataró" in final_answ


def test_5():
    english_command = "Will I be able to tend the clothes outside tomorrow?"
    tools = [
        {
            "name": "get_weather()",
            "description": "Call openweather api to retrieve the city's weather.",
            "args": {"city": "City Name", "date": "Literal: 'today' or 'tomorrow'"},
            "func": get_weather,
        }
    ]
    result = execute_tool_agent(english_command, tools)
    final_answ = result["messages"][-1].content
    assert "Mataró" in final_answ
