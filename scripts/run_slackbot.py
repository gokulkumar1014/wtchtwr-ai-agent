#!/usr/bin/env python3
"""
Start the Slack bot connected to the NL→SQL engine.

Usage:
    export SLACK_BOT_TOKEN="xoxb-..."
    export SLACK_SIGNING_SECRET="..."
    # Optional for Socket Mode:
    # export SLACK_APP_TOKEN="xapp-..."
    python scripts/run_slackbot.py
"""
# from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.slackbot import create_slack_app


def main() -> None:
    load_dotenv()
    app = create_slack_app()

    app_token = os.getenv("SLACK_APP_TOKEN")
    if app_token:
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        handler = SocketModeHandler(app, app_token)
        handler.start()
    else:
        port = int(os.getenv("SLACK_PORT", "3000"))
        app.start(port=port)


if __name__ == "__main__":
    main()
