from langchain_core.prompts import PromptTemplate
from typing import List
import models


def format_prompt(prompt) -> PromptTemplate:
    # TODO: format the input prompt by using the model specific instruction template
    # TODO: return a langchain PromptTemplate
    PROMPT_TEMPLATE = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
    {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return PromptTemplate.from_template(
        template=PROMPT_TEMPLATE,
        template_format="f-string"
    )

def format_chat_history(messages: List[models.Message]) -> str:
    # TODO:  implement format_chat_history to format 
    # the list of Message into a text of chat history.
    return "\n".join(
        [
            f"{m.role_type}: {m.message}" for m in messages
        ]
    )

def format_context(docs: List[str]):
    # TODO:  the output of the DataIndexer.search is a list of text, 
    # so we need to concatenate that list into a text that can fit into 
    # the rag_prompt_formatted. Implement format_context that takes a 
    # like of strings and returns the context as one string.
    return "\n\n".join(
        [
            f"Document {n}:\n{q}" for n, q in enumerate(docs, start=1)
        ]
    )

raw_prompt = "{question}"

# TODO: Create the history_prompt prompt that will capture the question and the conversation history. 
# The history_prompt needs a {chat_history} placeholder and a {question} placeholder.
history_prompt: str = """\
Given the following conversation provide a helpful answer to the follow up question.
Chat History:
{chat_history}
Follow Up question: {question}
helpful answer:"""

# TODO: Create the standalone_prompt prompt that will capture the question and the chat history
# to generate a standalone question. It needs a {chat_history} placeholder and a {question} placeholder,
standalone_prompt: str = """\
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

# TODO: Create the rag_prompt that will capture the context and the standalone question to generate
# a final answer to the question.
rag_prompt: str = """\
Answer the question based only on the following context:
{context}
Question: {standalone_question}
"""

# TODO: create raw_prompt_formatted by using format_prompt
raw_prompt_formatted = format_prompt(raw_prompt)
raw_prompt = PromptTemplate.from_template(raw_prompt)

# TODO: use format_prompt to create history_prompt_formatted
history_prompt_formatted: PromptTemplate = format_prompt(history_prompt)
# TODO: use format_prompt to create standalone_prompt_formatted
standalone_prompt_formatted: PromptTemplate = format_prompt(standalone_prompt)
# TODO: use format_prompt to create rag_prompt_formatted
rag_prompt_formatted: PromptTemplate = format_prompt(rag_prompt)





