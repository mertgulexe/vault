# Import necessary libraries
from dotenv import load_dotenv
from openai import OpenAI  # OpenAI's Python client library
import uuid  # For generating unique identifiers
import os


load_dotenv("/home/mert/Git/llm_bootcamp/week5/project/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Set the base URL for your vLLM API endpoint
# Replace [YOUR SPACE URL] with your actual Hugging Face Space URL
openai_api_base = "http://localhost:8000/v1"

# Initialize the OpenAI client
# We're using a custom base URL to point to our vLLM server
# The api_key is set to "NONE" as vLLM doesn't require authentication in this setup
client = OpenAI(
    api_key="OPENAI_API_KEY",
    base_url=openai_api_base,
)

# Create a chat completion request
stream = client.chat.completions.create(
    # Specify the model to use
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    
    # Define the conversation messages
    # Here, we're sending a single user message
    messages=[{
        "role": "user", 
        "content": "How many 'r's are there in the word 'strawberry'?"
    }],
    
    # Add a unique identifier to the query
    # This is a workaround for potential session management issues
    extra_query={
        'id': str(uuid.uuid4())
    },
    
    # Enable streaming of the response
    # This allows us to process the response as it's generated
    stream=True
)

# Process the streamed response
for chunk in stream:
    # Check if the chunk contains new content
    if chunk.choices[0].delta.content is not None:
        # Print the new content without a newline
        # This creates a continuous output as the response is generated
        print(chunk.choices[0].delta.content, end="")
print()

# Note: The script will continue to print until the entire response is received
# There's no explicit end to the for loop; it completes when the stream is exhausted