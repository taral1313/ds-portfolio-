from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # reads your .env file and loads the API key

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

response = llm.invoke("In one sentence, what is Retrieval-Augmented Generation?")

print("✅ OpenAI is connected!")
print(response.content)