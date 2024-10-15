# import streamlit as st
# import time
# from langserve import RemoteRunnable
from streamlit.runtime.scriptrunner import get_script_run_ctx
import streamlit as st
from pages.page_base import chat_interface

chat_title = "Simple Chat App"
url = "http://127.0.0.1:8001/simple/"
page_hash = get_script_run_ctx().page_script_hash

chat_interface(chat_title, page_hash, url)
