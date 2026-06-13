import google.generativeai as genai
genai.configure(api_key="AQ.Ab8RN6L8AlPLmkSQCAi-1ZED1Ak4PQM3lFM6KOVnT4azwRRDtQ")
model = genai.GenerativeModel("gemini-2.5-flash")
print("\nBot is ready...\n")
response= {
        "hi":"Hello",
        "hello":"Hi There",
        "who are you":"I am a Bot",
        "how are you":"I am good",
        "bye":"Goodbye"
        }
def responses(text):
    reply = response.get(text)
    if reply is None:
        gemini_response = model.generate_content(text)
        reply = gemini_response.text
    return reply
while True:
    user = input("You : ")
    if user.lower()=="end":
        break
    reply = responses(user.lower())
    print("Bot : "+reply)