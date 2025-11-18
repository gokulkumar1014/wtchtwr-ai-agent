from slack_bolt import App
import os

app = App(token=os.environ["SLACK_BOT_TOKEN"], signing_secret=os.environ["SLACK_SIGNING_SECRET"])

@app.event("app_mention")
def event_test(event, say):
    print("Event received:", event)
    say("✅ I received your mention!")

app.start(port=3000)
