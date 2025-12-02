"""
Here you will find the tools given to the Agent, specially to read and write
in the Google Drive.
"""


from langchain.tools import tool
from langchain_google_community import GoogleDriveLoader

@tool
def search_client_drive(query : str) -> str:

    loader = GoogleDriveLoader(
        folder_id="1f04r-mTCqEq0hYqTR5YK8ue0f7NRBT1e",
        token_path="token.pickle",
        credentials_path="credentials.json",
        recursive=True
    )

    docs = loader.load()


    relevant = []
    for doc in docs:
        if query.lower() in doc.page_content.lower():
            relevant.append(f"**{doc.metadata['name']}**:\n{doc.page_content[:300]}...")
    
    return "\n\n".join(relevant[:3]) if relevant else "No relevant info found."
