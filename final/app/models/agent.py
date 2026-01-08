from sqlalchemy import Column, String, Boolean
from app.core.database import Base


class Agent(Base):
    __tablename__ = "agents"

    agent_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    is_active = Column(Boolean, default=True)
