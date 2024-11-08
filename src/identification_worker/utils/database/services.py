import logging

import pandas as pd
from sqlalchemy import delete, select, func
from sqlalchemy.orm import Session

from .model import ReferenceImage

logger = logging.getLogger("database")


class ReferenceImageService:
    def __init__(self, engine):
        self.engine = engine

    def del_reference_images(self, organization_id: int):
        """Delete all Reference Image records associated with a specific organization from the database."""
        logger.info(f"Removing reference images for organization {organization_id}.")
        with Session(self.engine) as session:
            # remove old records
            stmt = delete(ReferenceImage).where(ReferenceImage.organization_id == organization_id)
            _ = session.execute(stmt)
            session.commit()

    def create_reference_images(self, organization_id: int, df: pd.DataFrame):
        """Create multiple Observation AI Prediction records for one observation."""
        assert "image_path" in df
        assert "class_id" in df
        assert "label" in df
        assert "embedding" in df
        with Session(self.engine) as session:
            stmt = delete(ReferenceImage).where(
                (ReferenceImage.organization_id == organization_id)
                & (ReferenceImage.image_path.in_(list(df["image_path"])))
            )
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

    def get_reference_images_count(self, organization_id: int) -> int:
        """Get the count of Reference Image records from the database."""
        try:
            # define count statement
            stmt = (
                select(func.count())
                .select_from(ReferenceImage)
                .where(ReferenceImage.organization_id == organization_id)
            )
            # execute statement
            with Session(self.engine) as session:
                count = session.execute(stmt).scalar()
                logger.info(
                    f"Total Reference Image records for organization {organization_id}: {count}"
                )
            return count
        except Exception as e:
            # Handle database connection or missing database error
            logger.error("Database does not exist or cannot be reached.")
            logger.debug(e)
            return 0

    def get_reference_images(
        self, organization_id: int, start: int = -1, end: int = -1, rows: tuple = ()
    ) -> pd.DataFrame:
        """Get dataframe with Reference Image records from the database.
        start - inclusive, end - exclusive
        """

        # define select statement
        stmt = select(ReferenceImage).where(ReferenceImage.organization_id == organization_id)
        if start >= 0 and end > 0:
            limit = end - start
            offset = start
            stmt = (
                select(ReferenceImage)
                .where(ReferenceImage.organization_id == organization_id)
                .limit(limit)
                .offset(offset)
            )
        elif rows:
            stmt = select(ReferenceImage).where(
                (ReferenceImage.organization_id == organization_id)
                & (ReferenceImage.order_idx.in_(rows))
            )

        # execute statement
        with Session(self.engine) as session:
            resp = session.execute(stmt).fetchall()
            reference_images = [x[0].data() for x in resp]
            reference_images = pd.DataFrame(reference_images)
            if len(reference_images) > 3:
                logger.info(f"Retrieved {len(reference_images)} Reference Image records.")
            return reference_images
