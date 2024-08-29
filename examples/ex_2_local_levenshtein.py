from Levenshtein import distance

from patronus import Client, Evaluator, EvaluationResult, simple_task


cli = Client()


class LevenshteinScorer(Evaluator):
    def __init__(self, pass_threshold: float):
        self.pass_threshold = pass_threshold
        super().__init__()

    def evaluate(self, evaluated_model_output: str, evaluated_model_gold_answer: str) -> EvaluationResult:
        max_len = max(len(x) for x in [evaluated_model_output, evaluated_model_gold_answer])
        score = 1
        if max_len > 0:
            score = 1 - (distance(evaluated_model_output, evaluated_model_gold_answer) / max_len)

        return EvaluationResult(
            score_raw=score,
            pass_=score >= self.pass_threshold,
            tags={"pass_threshold": str(self.pass_threshold)},
        )


cli.experiment(
    "LevenshteinScorer",
    data=[
        {
            "evaluated_model_input": "Foo",
            "evaluated_model_gold_answer": "Hi Foo",
        },
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "Hello Bar",
        },
        {
            "evaluated_model_input": "Bar",
            "evaluated_model_gold_answer": "eloh Bar!",
        },
        {
            "evaluated_model_input": "Hello Foo, how have you been?",
            "evaluated_model_gold_answer": "Hi Foo, nice to see you.",
        },
        {
            "evaluated_model_input": "Hi Bar, it's been a long time since we last talked.",
            "evaluated_model_gold_answer": "Hey Bar, how are you?",
        },
        {
            "evaluated_model_input": "Good day Baz, what have you been up to lately?",
            "evaluated_model_gold_answer": "Hi Baz, how have you been?",
        },
        {
            "evaluated_model_input": "Hello Qux, I hope you are doing well.",
            "evaluated_model_gold_answer": "Hello Qux, great to meet you.",
        },
        {
            "evaluated_model_input": "Hi Quux, how is everything going on your side?",
            "evaluated_model_gold_answer": "Hi Quux, hope you're well.",
        },
        {
            "evaluated_model_input": "Hey Corge, it's nice to catch up with you after so long.",
            "evaluated_model_gold_answer": "Hello Corge, what's new?",
        },
        {
            "evaluated_model_input": "Hi Grault, I hope you had a good weekend.",
            "evaluated_model_gold_answer": "Hello Grault, are you ready for the meeting?",
        },
        {
            "evaluated_model_input": "Hello Garply, how have you been these days?",
            "evaluated_model_gold_answer": "Hi Garply, any plans for the weekend?",
        },
        {
            "evaluated_model_input": "Hi Waldo, it's been great working with you.",
            "evaluated_model_gold_answer": "Hello Waldo, hope your day is going great.",
        },
        {
            "evaluated_model_input": "Hello Fred, how did the project go?",
            "evaluated_model_gold_answer": "Hi Fred, did you finish the project?",
        },
        {
            "evaluated_model_input": "Hey Plugh, what's new in your world?",
            "evaluated_model_gold_answer": "Hello Plugh, what are you working on today?",
        },
        {
            "evaluated_model_input": "Hi Xyzzy, welcome to the team.",
            "evaluated_model_gold_answer": "Hi Xyzzy, welcome to the new office!",
        },
        {
            "evaluated_model_input": "Hello Thud, how's everything on your side?",
            "evaluated_model_gold_answer": "Hello Thud, how are things on your end?",
        },
        {
            "evaluated_model_input": "Hi Zelda, great to have you with us.",
            "evaluated_model_gold_answer": "Hi Zelda, good to have you with us.",
        },
    ],
    task=simple_task(lambda input: f"Hi {input}"),
    evaluators=[LevenshteinScorer(pass_threshold=0.6)],
)
