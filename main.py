# TODO - Pull the data from a DB like Google Sheets
# TODO Tool to READ the DB

from agents import first_draft_agent


response = first_draft_agent.invoke({
    "role" : "user",
    "messages" : "Can you tell me more of what we have for Juanito Perez?"
})

for i,msg in enumerate(response["messages"].content):
    msg.pretty_print()