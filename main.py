from workflows.main.workflow import execute_main_workflow
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(
    filename="nabu_langgraph_agent.log", level=logging.INFO, filemode="w+"
)


def main():
    execute_main_workflow("Posa música dels Amics de les Arts")


if __name__ == "__main__":
    main()
