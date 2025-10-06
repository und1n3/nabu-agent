# Nabu Agent

A voice-controlled AI agent system that processes audio commands and executes actions across multiple platforms including Spotify, Home Assistant, and web search. The agent uses speech-to-text, translation, and classification to route commands to the appropriate handler.

## Features

- **Speech-to-Text**: Converts audio input to text using Faster Whisper
- **Multi-language Support**: Automatic language detection and translation to English
- **Smart Command Classification**: Routes commands to appropriate handlers:
  - **Spotify Integration**: Play music, artists, albums, playlists, and radio
  - **Home Assistant**: Control smart home devices
  - **Web Search**: Internet search via SearxNG
  - **Party Commands**: Pre-established custom commands with witty responses
- **LangChain Integration**: Uses LangChain for LLM orchestration and MCP adapters for Home Assistant
- **Workflow Visualization**: Built with LangGraph for complex workflow management

## Architecture

The system uses a workflow-based architecture with the following nodes:

1. **STT (Speech-to-Text)**: Transcribes audio input using Faster Whisper
2. **Translator**: Detects language and translates to English
3. **Enrouting Question**: Classifies the command type
4. **Command Handlers**:
   - Pre-established commands (party mode)
   - Internet search
   - Spotify command (with sub-workflow)
   - Home Assistant command
5. **Finish Action**: Prepares and translates the final response

## Installation

### Prerequisites

- Python 3.12 or higher
- UV package manager (recommended) or pip

### Install Dependencies

Using UV:
```bash
uv sync
```

Using pip:
```bash
pip install -e .
```

For development:
```bash
uv sync --dev
```

## Configuration

Create a `.env` file in the project root with the following environment variables:

### Required Environment Variables

```bash
# LangChain Configuration
LANGCHAIN_API_KEY=...              # LangChain API key for tracing
LANGCHAIN_TRACING_V2=true          # Enable LangChain tracing
LANGCHAIN_PROJECT=nabu-agent       # Project name for LangChain

# LLM Configuration
LLM_BASE_URL=...                   # Base URL for the LLM API
LLM_API_KEY=...                    # API key for LLM access
LLM_MODEL=Qwen3-4B                 # LLM model to use (e.g., Qwen3-4B)

# Faster Whisper (STT) Configuration
FASTER_WHISPER_MODEL=...           # Whisper model size (e.g., base, small, medium, large)
FASTER_WHISPER_USE_CUDA=false      # Set to 'true' to use CUDA acceleration

# Search Configuration
SEARX_HOST=...                     # SearxNG instance URL for web searches

# Spotify Configuration
SPOTIPY_CLIENT_ID=...              # Spotify API client ID
SPOTIPY_CLIENT_SECRET=...          # Spotify API client secret
SPOTIPY_REDIRECT_URI=https://127.0.0.1:1234  # OAuth redirect URI

# Home Assistant Configuration
HA_TOKEN=...                       # Home Assistant long-lived access token
HA_URL=...                         # Home Assistant instance URL (e.g., http://homeassistant.local:8123)
```

### Environment Variable Details

- **LANGCHAIN_API_KEY**: Get from [LangSmith](https://smith.langchain.com/)
- **LLM_BASE_URL**: OpenAI-compatible API endpoint (e.g., local Ollama, OpenAI, etc.)
- **FASTER_WHISPER_MODEL**: Choose from: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`
- **SEARX_HOST**: URL to your SearxNG instance (self-hosted or public)
- **Spotify credentials**: Get from [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- **HA_TOKEN**: Generate from Home Assistant: Profile → Security → Long-Lived Access Tokens

## Usage

### Command Line Interface

Run the agent with an audio file:

```bash
uv run nabu-agent /path/to/audio/file.wav
```

### Programmatic Usage

```python
import asyncio
from nabu_agent import execute_main_workflow

# Read audio file
with open("audio.wav", "rb") as f:
    audio_data = f.read()

# Execute workflow
result = asyncio.run(execute_main_workflow(audio_data))
print(result)

# Generate workflow visualization
result = asyncio.run(execute_main_workflow(audio_data, graph=True))
# This creates graph.png and full_graph.png
```

## Command Examples

### Spotify Commands
- "Play music by The Beatles"
- "Play the song Bohemian Rhapsody"
- "Play the album Dark Side of the Moon"
- "Play my Discover Weekly playlist"
- "Play radio for Pink Floyd"

### Home Assistant Commands
- "Turn on the living room lights"
- "What devices do I have?"
- "Set the thermostat to 72 degrees"

### Internet Search
- "What's the weather today?"
- "Search for Python tutorials"
- "What's the latest news?"

### Party Commands
- "Tick-tock" (starts a countdown with an ominous sentence)

## Development

### Project Structure

```
nabu-agent/
├── src/nabu_agent/
│   ├── main.py                 # Entry point
│   ├── workflows/
│   │   ├── main/              # Main workflow
│   │   │   ├── workflow.py
│   │   │   ├── nodes.py
│   │   │   └── state.py
│   │   └── spotify_agent/     # Spotify sub-workflow
│   │       ├── workflow.py
│   │       └── nodes.py
│   ├── tools/
│   │   ├── agents.py          # LLM agents (STT, classifier, translator)
│   │   ├── spotify.py         # Spotify integration
│   │   └── web_loader.py      # Web search
│   ├── utils/
│   │   └── schemas.py         # Pydantic models
│   └── data/
│       └── preestablished_commands.py
├── tests/
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
pytest
```

### Adding Pre-established Commands

Edit `src/nabu_agent/data/preestablished_commands.py`:

```python
party_commands = {
    "tick-tock": "start a countdown and tell an ominous sentence.",
    "your-command": "description of what this command does"
}
```

## Logging

Logs are written to `nabu_agent_agent.log` in the current directory. Check this file for detailed execution information and debugging.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

## License

See LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Spotify device not found**: Ensure you have an active Spotify device. The system looks for a device named "librespot" with specific ID. Update `DEVICE_NAME` and `DEVICE_ID` in `src/nabu_agent/tools/spotify.py` if needed.

2. **Audio format errors**: Ensure your audio files are in a compatible format (WAV, MP3, etc.)

3. **Home Assistant connection**: Verify your `HA_URL` includes the full URL with protocol and port (e.g., `http://homeassistant.local:8123`)

4. **Language detection issues**: The system translates all commands to English. If you're getting incorrect results, check the `original_language` in the logs.

## Requirements

Core dependencies:
- `faster-whisper>=1.2.0` - Speech-to-text
- `langchain-community>=0.3.26` - LangChain community tools
- `langchain-mcp-adapters>=0.1.10` - MCP protocol adapters
- `langchain-openai>=0.3.33` - OpenAI LLM integration
- `langgraph>=0.4.8` - Workflow graph builder
- `pydantic>=2.11.7` - Data validation
- `python-dotenv>=1.1.1` - Environment management
- `spotipy>=2.25.1` - Spotify API client
