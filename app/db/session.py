import os

from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://cia_user:cia_password@localhost:5432/customer_intelligence",
)