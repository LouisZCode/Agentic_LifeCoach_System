# TODO - Create an Undefined Agent expert
# TODO -  Able to make the Agent create a new Client Folder form the Survey
# TODO - Able to create a Discovery_Prep .txt from the Survey IN the Clients File

# Logic, Maris: Hey! I got a new answer form the Survey, here it is:
# Outout: I created a new File called [Client_Name], Saved the survey_answer document in it, and created a Discovery Prep from the
#Template into that Folder. 

from agents import first_draft_agent

response = first_draft_agent.invoke({
    "role": "user",
    "messages": "Can you summarize in one sentence to me the session with Juanito?"
})

