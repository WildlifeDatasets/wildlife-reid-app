import logging

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from commonlib.error_handling import retry

from .model.base import Base


logger = logging.getLogger("database")


class DbConnection:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = self._connect(db_url)

        # create services with methods for reading/writing from the db
        self.image = ImageService(self.engine)
        self.metadata = MetadataService(self.engine)
        self.observation = ObservationService(self.engine)
        self.body_part = BodyPartService(self.engine)
        self.species = SpeciesService(self.engine)
        self.user = UserService(self.engine)
        self.admin = AdminService(self.engine)

    @retry(
        num_retries=5,
        exceptions=(OperationalError,),
        delay=10.0,
        init_message="Creating database connection.",
        retry_message="Failed to connect to the Postgres database. Retrying...",
        fail_message="Failed to connect to the Postgres database",
    )
    def _connect(self, db_url):
        # create engine
        # note: future=True ensures that the latest SQLAlchemy 2.0-style APIs is used
        # note: the engine does not establish the first actual DBAPI connection
        #       until the Engine.connect() method is called
        engine = create_engine(db_url, future=True)
        Base.metadata.create_all(engine)
        return engine
