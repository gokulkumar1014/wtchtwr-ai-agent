# scripts/test_slackbot_local.py

from agent.slack.bot import handle_question

def test_local_question():
    question = "What is the average occupancy in Manhattan?"
    result = handle_question(question)
    print("\n=== RESULT ===")
    print(result)
    print("=================")

if __name__ == "__main__":
    test_local_question()
