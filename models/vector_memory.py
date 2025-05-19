from sqlalchemy import Column, Integer, JSON, Text, TIMESTAMP
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text

Base = declarative_base()

class VectorEntry(Base):
    __tablename__ = "vector_memory"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    embedding   = Column(Vector(1536), nullable=False)
    meta        = Column("metadata", JSON, nullable=False)
    content     = Column(Text, nullable=False)
    created_at  = Column(
                    TIMESTAMP(timezone=True),
                    server_default=text("NOW()")
                 )
