import numpy as np


class ClauseIndex:
    def __init__(self, client):
        self.client = client
        self.records = []  # list of dicts: {contract_id, clause_type, text}
        self._vectors = None

    def build(self, results: list[dict]):
        """results: the list of per-contract dicts produced by the pipeline."""
        for row in results:
            for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
                text = row.get(clause_type, "")
                if text:
                    self.records.append(
                        {
                            "contract_id": row["contract_id"],
                            "clause_type": clause_type,
                            "text": text,
                        }
                    )

        vectors = [self.client.embed(r["text"]) for r in self.records]
        self._vectors = np.array(vectors)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._vectors is None or len(self.records) == 0:
            return []

        q_vec = np.array(self.client.embed(query))
        sims = self._cosine_sim(self._vectors, q_vec)
        top_idx = np.argsort(-sims)[:top_k]

        results = []
        for i in top_idx:
            record = dict(self.records[i])
            record["score"] = float(sims[i])
            results.append(record)
        return results

    @staticmethod
    def _cosine_sim(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        norm_matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        norm_vector = vector / (np.linalg.norm(vector) + 1e-8)
        return norm_matrix @ norm_vector
