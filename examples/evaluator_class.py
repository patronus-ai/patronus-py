from transformers import BertTokenizer, BertModel
import numpy as np
from patronus import Client, Evaluator, EvaluationResult, Row

client = Client()


class BERTScore(Evaluator):
    def __init__(self, pass_threshold: float):
        self.pass_threshold = pass_threshold
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased")

        super().__init__()

    def evaluate(self, row: Row) -> EvaluationResult:
        # Tokenize text
        output_toks = self.tokenizer(row.evaluated_model_output, return_tensors="pt", padding=True, truncation=True)
        gold_answer_toks = self.tokenizer(
            row.evaluated_model_gold_answer, return_tensors="pt", padding=True, truncation=True
        )

        # Obtain embeddings from BERT model
        output_embeds = self.model(**output_toks).last_hidden_state.mean(dim=1).detach().numpy()
        gold_answer_embeds = self.model(**gold_answer_toks).last_hidden_state.mean(dim=1).detach().numpy()

        # Calculate cosine similarity
        score = np.dot(output_embeds, gold_answer_embeds.T) / (
            np.linalg.norm(output_embeds) * np.linalg.norm(gold_answer_embeds)
        )

        return EvaluationResult(
            score_raw=score,
            pass_=score >= self.pass_threshold,
            tags={"pass_threshold": str(self.pass_threshold)},
        )


client.experiment(
    "Tutorial",
    dataset=[
        {
            "evaluated_model_input": "Translate 'Goodbye' to Spanish.",
            "evaluated_model_output": "Hasta luego",
            "evaluated_model_gold_answer": "Adiós",
        },
        {
            "evaluated_model_input": "Summarize: 'The quick brown fox jumps over the lazy dog'.",
            "evaluated_model_output": "Quick brown fox jumps over dog",
            "evaluated_model_gold_answer": "The quick brown fox jumps over the lazy dog",
        },
    ],
    evaluators=[BERTScore(pass_threshold=0.8)],
    experiment_name="BERTScore Output Label Similarity",
)
