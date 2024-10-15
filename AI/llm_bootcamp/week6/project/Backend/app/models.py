from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    messages = relationship("Message", back_populates="user")

# TODO: Implement the Message SQLAlchemy model. Message should have a primary key, 
# a message attribute to store the content of messages, a type, AI or Human, 
# depending on if it is a user question or an AI response, a timestamp to 
# order by time and a user attribute to get the user instance associated 
# with the message. We also need a user_id that will use the User.id 
# attribute as a foreign key.
class Message(Base):
    __tablename__ = "messages"
    message_id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )
    message = Column(String, index=False)
    role_type = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    user_id = Column(Integer, ForeignKey("users.user_id"), autoincrement=True)
    user = relationship("User", back_populates="messages")
