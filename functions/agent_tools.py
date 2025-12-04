"""
Here you will find the tools given to the Agent, specially to read and write
in the Google Drive.
"""
from config import TEST_FOLDER_ID

from langchain.tools import tool
from langchain_google_community import GoogleDriveLoader
from .helper_functions import _get_drive_service, get_folder_id_by_name, load_drive_folder
from config import TEST_FOLDER_ID
from langchain.tools import tool

# Configuration
ACTIVE_CLIENTS_ROOT_ID = TEST_FOLDER_ID 


@tool
def read_client_folder(client_name: str):
    """
    Finds a client's folder by name inside the 'Active' directory 
    and returns the content of their documents.
    Useful for answering questions about a specific client.
    """
    print(f"🕵️  Agent is searching for client folder: '{client_name}'...")
    
    # 1. Connect (using your existing setup)
    service = _get_drive_service("credentials.json")
    
    # 2. Find the specific client's folder ID
    client_folder_id = get_folder_id_by_name(service, client_name, parent_id=ACTIVE_CLIENTS_ROOT_ID)
    
    if not client_folder_id:
        return f"Error: Could not find an active client folder named '{client_name}'."
        
    # 3. Load the documents from THAT specific folder
    print(f"📖  Found folder {client_folder_id}. Reading documents...")
    docs = load_drive_folder(client_folder_id, "credentials.json", service)
    
    if not docs:
        return f"Found folder for '{client_name}', but it was empty."
        
    # 4. Return text to the Agent
    # We combine all docs into one big text string for the LLM to read
    full_text = "\n\n".join([f"--- Doc: {d.metadata['title']} ---\n{d.page_content}" for d in docs])
    return full_text
