from sqlalchemy import Column, String, Boolean, JSON, Integer
from app.core.database import Base


class WebhookConfig(Base):
    __tablename__ = "webhooks"

    id = Column(Integer, primary_key=True, index=True)
    webhook_url = Column(String)
    agent_id = Column(String, nullable=True)
    events = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
