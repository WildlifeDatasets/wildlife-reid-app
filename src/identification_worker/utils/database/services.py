import logging

import pandas as pd
from sqlalchemy import delete, select
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
            stmt = delete(ReferenceImage).where(
                ReferenceImage.organization_id == organization_id and ReferenceImage.image_path in df["image_path"])

            # stmt = delete(ReferenceImage).where(
            #     (ReferenceImage.organization_id == organization_id) & (ReferenceImage.image_path.in_(list(df["image_path"]))))
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

    def get_reference_images(self, organization_id: int) -> pd.DataFrame:
        """Get dataframe with Reference Image records from the database."""
        # define select statement
        stmt = select(ReferenceImage).where(ReferenceImage.organization_id == organization_id)

        # execute statement
        with Session(self.engine) as session:
            resp = session.execute(stmt).fetchall()
            reference_images = [x[0].data() for x in resp]
            reference_images = pd.DataFrame(reference_images)
            logger.info(f"Retrieved {len(reference_images)} Reference Image records.")
            return reference_images
