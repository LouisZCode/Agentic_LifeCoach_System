# TODO - Pull the data from a DB like Google Sheets
# TODO Tool to READ the DB

from agents import first_draft_agent

response = first_draft_agent.invoke({
    "role": "user",
    "messages": "Can you summarize in one sentence to me the session with Juanito?"
})

# Show what happened
print("\n" + "="*50)
for msg in response["messages"]:
    if msg.type == "ai":
        # Check if it used tools
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"🔧 Agent used tool: {msg.tool_calls[0]['name']}")
        # Show the final answer
        elif hasattr(msg, 'content'):
            if isinstance(msg.content, str):
                print(f"\n💬 Answer:\n{msg.content}")
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if block.get('type') == 'text':
                        print(f"\n💬 Answer:\n{block['text']}")
print("="*50 + "\n")