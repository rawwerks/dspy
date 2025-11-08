import dspy
from dspy.clients.cli_lm import CLILM


class SimpleMath(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

lm = CLILM("claude -p --dangerously-skip-permissions")
dspy.configure(lm=lm)

predictor = dspy.Predict(SimpleMath)
question = "What is 2 + 2?"
result = predictor(question=question)

print(f"Question: {question}")
print(f"Claude answer: {result.answer}")
print(lm.history)
