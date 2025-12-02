# test_folder_access.py
from langchain_google_community import GoogleDriveLoader


FOLDER_ID = "1f04r-mTCqEq0hYqTR5YK8ue0f7NRBT1e"



loader = GoogleDriveLoader(
    folder_id=FOLDER_ID,
    service_account_key="credentials.json",
    recursive=True,
    file_types=["document", "pdf", "sheet"]
)

print("🔍 Attempting to load documents...")
docs = loader.load()

print(f"✅ Found {len(docs)} document(s)!")
for doc in docs[:3]:  # Show first 3
    print(f"  📄 {doc.metadata.get('name', 'Unknown')}")