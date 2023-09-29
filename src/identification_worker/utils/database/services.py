import logging

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.orm import Session

from .model import ReferenceImage

logger = logging.getLogger("database")


class ReferenceImageService:
    def __init__(self, engine):
        self.engine = engine

    def create_reference_images(self, organization_id: int, df: pd.DataFrame):
        """Create multiple Observation AI Prediction records for one observation."""
        assert "image_path" in df
        assert "class_id" in df
        assert "label" in df
        assert "embedding" in df
        with Session(self.engine) as session:
            # remove old records
            stmt = delete(ReferenceImage).where(ReferenceImage.organization_id == organization_id)
            _ = session.execute(stmt)

            # add new records
            df = df[["image_path", "class_id", "label", "embedding"]].copy()
            df["organization_id"] = organization_id
            num_records = df.to_sql(
                name=ReferenceImage.__tablename__,
                con=session.connection(),  # self.engine,
                schema=None,
                index=False,
                if_exists="append",  # append records if the table exists
                method="multi",
            )
            session.commit()
            logger.info(f"Added {num_records} Observation AI Prediction records.")
