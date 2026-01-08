from sqlalchemy import Column, String, Boolean, DateTime
from datetime import datetime
from app.core.database import Base


class PhoneNumber(Base):
    __tablename__ = "phone_numbers"

    id = Column(String, primary_key=True)
    phone_number = Column(String, unique=True, index=True)
    agent_id = Column(String, nullable=True)
    label = Column(String, nullable=True)
    provider = Column(String, default="twilio")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
