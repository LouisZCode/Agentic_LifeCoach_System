# TODO - Pull the data from a DB like Google Sheets
# TODO Tool to READ the DB

from agents import first_draft_agent



response = first_draft_agent.invoke({
    "role" : "user",
    "messages" : "Hello, what can you do?"
})

for i,msg in enumerate(response["messages"]):
    msg.pretty_print()