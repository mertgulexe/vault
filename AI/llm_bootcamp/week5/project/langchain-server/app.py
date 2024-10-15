from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# Create a ChatOpenAI instance with custom configuration
chat = ChatOpenAI(
    model="meta-llama/Meta-Llama-3-8B-Instruct", 
    openai_api_base="http://0.0.0.0:8000/v1", 
    openai_api_key="[YOUR HUGGING FACE TOKEN]",
    max_tokens=512,
)

# Create a message
message = HumanMessage(content="How many 'r's are there in the word 'strawberry'?")

# Generate the response
response = chat.stream([message])
for chunk in response:
    print(chunk.content, end="", flush=True)
print()