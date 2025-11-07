from typing import Literal

import openmeteo_requests
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from langchain.tools import tool

load_dotenv()
WEATHER_CODES = {
    0: "Clear",
    1: "Mostly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Icy Fog",
    51: "Light Drizzle",
    53: "Drizzle",
    55: "Heavy Drizzle",
    56: "Light Freezing Drizzle",
    57: "Freezing Drizzle",
    61: "Light Rain",
    63: "Rain",
    65: "Heavy Rain",
    66: "Light Freezing Rain",
    67: "Freezing Rain",
    71: "Light Snow",
    73: "Snow",
    75: "Heavy Snow",
    77: "Snow Grains",
    80: "Light Showers",
    81: "Showers",
    82: "Heavy Showers",
    85: "Light Snow Showers",
    86: "Snow Showers",
    95: "Thunderstorm",
    96: "Light T-storm w/ Hail",
    99: "T-storm w/ Hail",
}


def get_todays_forecast(lon: float, lat: float) -> str:
    openmeteo = openmeteo_requests.Client()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "precipitation",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
        ],
        "timezone": "Europe/Berlin",
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    current = response.Current()
    current_temperature_2m = current.Variables(0).Value()
    current_precipitation = current.Variables(1).Value()
    current_weather_code = current.Variables(2).Value()
    current_cloud_cover = current.Variables(3).Value()
    current_wind_speed_10m = current.Variables(4).Value()
    summary = f"""
    temperature: {current_temperature_2m}
    precipitation: {current_precipitation}
    weather_code: {WEATHER_CODES[current_weather_code]}
    cloud_cover: {current_cloud_cover}
    wind_speed: {current_wind_speed_10m}
    """
    return summary


def get_tomorrows_forecast(lon: float, lat: float):
    openmeteo = openmeteo_requests.Client()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "precipitation_probability_max",
            "precipitation_hours",
        ],
        "timezone": "Europe/Berlin",
        "forecast_days": 3,
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    daily_weather_code = daily.Variables(0).ValuesAsNumpy().tolist()
    daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy().tolist()
    daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy().tolist()
    daily_precipitation_sum = daily.Variables(3).ValuesAsNumpy().tolist()
    daily_precipitation_probability_max = daily.Variables(4).ValuesAsNumpy().tolist()

    summary = f"""
    temperature_max: {daily_temperature_2m_max[1]}
    temperature_min:{daily_temperature_2m_min[1]}
    precipitation: {daily_precipitation_sum[1]}
    precipitation_probability:{daily_precipitation_probability_max[1]}
    weather_code: {WEATHER_CODES[daily_weather_code[1]]}
    """
    return summary


def get_coords(city_name):
    geolocator = Nominatim(user_agent="city_locator")
    location = geolocator.geocode(city_name)
    if location:
        return {"lat": location.latitude, "lon": location.longitude}
    else:
        raise ValueError(f"Could not find coordinates for city: {city_name}")


@tool
def get_weather(city: str = "Mataró", date: Literal["today", "tomorrow"] = "today"):
    """
    Get the weather forecast
    args:
    - city (str) : The city name.
    - date (str) : A literal for the day to get the weather for, either "today" or "tomorrow"
    """

    # get city's lat/lon
    coords = get_coords(city)

    # get summary forecast:
    if date == "tomorrow":
        return get_tomorrows_forecast(lon=coords["lon"], lat=coords["lat"])

    return get_todays_forecast(lon=coords["lon"], lat=coords["lat"])
