"""
Here you will find the tools given to the Agent, specially to read and write
in the Local Files.
"""
from sqlalchemy.sql.coercions import TruncatedLabelImpl
from config import TEMPLATES_PATH, ACTIVE_CLIENTS_PATH
from langchain.tools import tool
from pathlib import Path


@tool(
    "read_folder",
    parse_docstring=True,
    description="reads the existing folder names with client names"
)
def read_folder(path : str) -> str:
    """
    Description:
        Read the folder as a list of the names on 1 level.

    Args: 
        path (str): this is the path where the folders you are searching for are.

    returns:
        The exact and correct names of the clients

    raises:
        Error if the path is empty

    """
    active_path = Path(path)
    clients = [folder.name for folder in active_path.iterdir() if folder.is_dir()]

    return clients

@tool(
    "read_template",
    parse_docstring=True,
    description="reads the desired Template with desired structure"
)
def read_template():
    """
    Read the folder as a list of the names on 1 level.
    """
    active_path = Path(ACTIVE_CLIENTS_PATH)
    clients = [folder.name for folder in active_path.iterdir() if folder.is_dir()]

    return clients

@tool
def create_summary(path: str):
    """
    Read the folder as a list of the names on 1 level.
    """
    pass

@tool
def create_homework(path: str):
    """
    Read the folder as a list of the names on 1 level.
    """
    pass