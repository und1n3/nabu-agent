import argparse
import asyncio
import logging

from dotenv import load_dotenv

from .workflows.main.workflow import execute_main_workflow

load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(filename="nabu_agent_agent.log", level=logging.INFO, filemode="w+")


def app():
    # All the logic of argparse goes in this function
    parser = argparse.ArgumentParser(description="Say hi.")
    parser.add_argument(
        "input",
        type=str,
        help="Input",
    )
    args = parser.parse_args()
    with open(args.input, "rb") as f:
        res = asyncio.run(execute_main_workflow(f.read()))
    logger.info(res)


if __name__ == "__main__":
    app()
