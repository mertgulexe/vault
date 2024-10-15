import models 
import schemas
from sqlalchemy.orm import Session


def get_or_create_user(db: Session, username: str) -> models.User:
    U = models.User
    user = db.query(U).filter(U.username == username).first()
    if not user:
        user = U(username=username)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def add_message(db: Session, message: schemas.MessageBase, username: str):
    # TODO:  Implement the add_message function. It should:
    # - get or create the user with the username
    # - create a models.Message instance
    # - pass the retrieved user to the message instance
    # - save the message instance to the database
    user = get_or_create_user(db=db, username=username)
    message_instance = models.Message()
    message_instance.message_id = message.message_id
    message_instance.message = message.message
    message_instance.role_type = message.role_type
    message_instance.timestamp = message.timestamp
    message_instance.user_id = message.user_id
    message_instance.user = user
    db.add(message_instance)
    db.commit()
    db.refresh(message_instance)

def get_user_chat_history(db: Session, username: str):
    U = models.User
    user = db.query(U).filter(U.username == username).first()
    return user.messages if user is not None else []
