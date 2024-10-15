from streamlit.runtime.scriptrunner import get_script_run_ctx
from pages.page_base import chat_interface


chat_title = "Filtered RAG Chat App"
url = "http://127.0.0.1:8001/filtered_rag/"
page_hash = get_script_run_ctx().page_script_hash

chat_interface(chat_title, page_hash, url)