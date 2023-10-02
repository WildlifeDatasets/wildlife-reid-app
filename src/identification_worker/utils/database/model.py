import logging
from datetime import datetime
from enum import Enum
from typing import List

from sqlalchemy import JSON, Column, DateTime, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

logger = logging.getLogger("database")


class CustomBase:
    date_created = Column(DateTime(timezone=False), server_default=func.now())
    date_updated = Column(DateTime(timezone=False), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return self.to_str()

    @property
    def columns(self) -> list:
        """Get list of all columns in the table."""
        return [x.name for x in self.__table__.columns]

    def to_str(self, columns: List[str] = None) -> str:
        """Convert database record to string.

        :param columns: List of columns to include in the output string.
        :return: String description of a database record.
        """

        def parse_value(v):
            if isinstance(v, datetime):
                v = v.strftime("%d-%m-%Y_%H:%M:%S.%f")
            elif isinstance(v, Enum):
                v = v.name
            v = f"{v!r}"  # to string by calling __repr__
            v = v if len(v) <= 50 else "..."  # omit long values, e.g. UUID("...") has 44 characters
            return v

        if columns is None:
            columns = self.columns
            # reorder columns
            if "id" in columns:
                columns = ["id"] + [x for x in columns if x != "id"]
            columns = [x for x in columns if x not in ("date_created", "date_updated")] + [
                "date_created",
                "date_updated",
            ]
        columns_str = ", ".join([f"{x}={parse_value(getattr(self, x))}" for x in columns])
        return f"{self.__class__.__name__}({columns_str})"

    def to_dict(self, columns: List[str] = None) -> dict:
        """Convert database record to the dictionary.

        :param columns: List of columns to include in the output dict.
        :param exclude_internal: If true, exclude internal columns in the output dict.
        :return: Dictionary with database record information.
        """
        if columns is None:
            columns = self.columns
        return {x: getattr(self, x) for x in columns}

    def data(self) -> dict:
        """Get a dictionary with data from the main database table and related tables."""
        return self.to_dict()


Base = declarative_base(cls=CustomBase)


class ReferenceImage(Base):
    __tablename__ = "reference_image"
    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer)
    image_path = Column(String)
    class_id = Column(Integer)
    label = Column(String)
    embedding = Column(JSON)
