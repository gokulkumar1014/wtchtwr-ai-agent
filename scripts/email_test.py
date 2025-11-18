import sys
from pathlib import Path

# Add project root to PYTHONPATH dynamically
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.emailer import send_email

send_email(
    recipient="gokulkumar0639@gmail.com",
    subject="HOPE AI Email Test",
    body="This is a test email sent from HOPE AI backend!",
)
