"""Loads .env before anything else reads os.environ. Import this first in main.py."""
from dotenv import load_dotenv
load_dotenv()
