from datetime import datetime
from typing import Dict, Any, List
from langchain_core.callbacks import BaseCallbackHandler
import schemas
import crud


class LogResponseCallback(BaseCallbackHandler):

    def __init__(self, user_request: schemas.UserRequest, db):
        super().__init__()
        self.user_request = user_request
        self.db = db

    def on_llm_end(self, outputs: Dict[str, Any], **kwargs: Any):
        """Run when llm ends running."""
        # TODO: The function on_llm_end is going to be called when the LLM stops sending 
        # the response. Use the crud.add_message function to capture that response.
        message = schemas.MessageBase(
            message=outputs.generations[0][0].text,
            role_type="AI",
            timestamp=datetime.now(),
            user=self.user_request.username
        )
        crud.add_message(
            db=self.db,
            message=message,
            username=self.user_request.username
        )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        for prompt in prompts:
            print("USER PROMPT:\n", prompt)