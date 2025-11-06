import logging
import subprocess
from typing import Optional
from langchain.tools import tool

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

from ..utils.schemas import SpotifyType

load_dotenv()
logger = logging.getLogger(__name__)
scope = [
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "app-remote-control",
    "streaming",
    "user-read-playback-position",
    "user-top-read",
    "user-read-recently-played",
    "user-library-read",
]

DEVICE_NAME = "librespot"
DEVICE_ID = "7c28ab8a5c9512e4266ac7cb756312c82ee43d7e"


def init_spotify() -> spotipy.Spotify:
    spotify_client = spotipy.Spotify(
        auth_manager=SpotifyOAuth(scope=scope, cache_path="/.cache")
    )
    device_active = False
    for device in spotify_client.devices()["devices"]:
        if device["id"] == "7c28ab8a5c9512e4266ac7cb756312c82ee43d7e":
            device_active = True
            logger.info("librespot device already active")
            break
    if not device_active:
        logger.info("enabling librespot device")
        subprocess.run(["./spotify-connect", "192.168.0.13", "5577"])

    return spotify_client


def play_music(
    spotify_client: spotipy.Spotify,
    context_uri: Optional[str] = None,
    uris: Optional[str] = None,
) -> None:
    if context_uri:
        spotify_client.start_playback(device_id=DEVICE_ID, context_uri=context_uri)
    else:
        spotify_client.start_playback(device_id=DEVICE_ID, uris=[uris])


def search_music(
    spotify_client: spotipy.Spotify, criteria_type: SpotifyType, query: str
):
    if criteria_type == SpotifyType.RADIO:
        criteria_type = SpotifyType.PLAYLIST
        query = "this%20is%20" + query
    else:
        criteria_type = criteria_type.value

    print(f"______________ {criteria_type} ____ {query}")
    try:
        result = spotify_client.search(q=query, type=[criteria_type], limit=2)
        result_id = result[criteria_type + "s"]["items"][0]["uri"]
    except:
        result = spotify_client.search(q=query, type=["track"], limit=2)
        result_id = result["tracks"]["items"][0]["uri"]

    return result_id


@tool
def pause_music():
    """
    Pause the current Spotify playback on the configured device.

    If no playback is active, this function has no effect.

    Raises:
        spotipy.SpotifyException: If the API request fails.
    """
    spotify_client = init_spotify()
    playback = spotify_client.current_playback()
    if playback and playback["device"]:
        spotify_client.pause_playback(device_id=DEVICE_ID)


@tool
def next_song():
    """
    Skip to the next track in the current Spotify playback queue.

    If no track is currently playing, the function has no effect.

    Raises:
        spotipy.SpotifyException: If the API request fails.
    """
    spotify_client = init_spotify()
    spotify_client.next_track(device_id=DEVICE_ID)


@tool
def previous_song():
    """
    Go back to the previous track in the current Spotify playback queue.

    If there is no previous track, this function has no effect.

    Raises:
        spotipy.SpotifyException: If the API request fails.
    """
    spotify_client = init_spotify()
    spotify_client.previous_track(device_id=DEVICE_ID)


@tool
def volume_up(sum_vol=10):
    """
    Increase the Spotify playback volume by a specified amount.

    Args:
        sum_vol (int, optional): The number of percentage points to increase
            the volume by. Defaults to 10.

    Notes:
        - The volume will not exceed 100%.
        - If no active playback is found, a warning is logged.

    Raises:
        spotipy.SpotifyException: If the API request fails.
    """
    spotify_client = init_spotify()
    playback = spotify_client.current_playback()
    if playback and playback["device"]:
        current_volume = playback["device"]["volume_percent"]
        new_volume = min(current_volume + sum_vol, 100)
        spotify_client.volume(new_volume, device_id=DEVICE_ID)
        logger.info(f"Volume increased to {new_volume}%")
    else:
        logger.warning("No active playback device found.")


@tool
def volume_down(sum_vol=10):
    """
    Decrease the Spotify playback volume by a specified amount.

    Args:
        sum_vol (int, optional): The number of percentage points to decrease
            the volume by. Defaults to 10.

    Notes:
        - The volume will not go below 0%.
        - If no active playback is found, a warning is logged.

    Raises:
        spotipy.SpotifyException: If the API request fails.
    """
    spotify_client = init_spotify()
    playback = spotify_client.current_playback()
    if playback and playback["device"]:
        current_volume = playback["device"]["volume_percent"]
        new_volume = max(current_volume - sum_vol, 0)
        spotify_client.volume(new_volume, device_id=DEVICE_ID)
        logger.info(f"Volume decreased to {new_volume}%")
    else:
        logger.warning("No active playback device found.")
