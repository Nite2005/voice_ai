from sqlalchemy import Column, String, JSON, DateTime
from datetime import datetime
from app.core.database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(String, primary_key=True, index=True)
    agent_id = Column(String)
    phone_number = Column(String)
    status = Column(String)
    dynamic_variables = Column(JSON)
    call_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
