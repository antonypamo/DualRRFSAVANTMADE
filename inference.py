import argparse
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer, util


def load_model(model_path: Path) -> SentenceTransformer:
    """Load a SentenceTransformer model from the given path."""
    resolved_path = model_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {resolved_path}")
    return SentenceTransformer(str(resolved_path))


def compute_similarity(model: SentenceTransformer, source: str, target: str) -> float:
    """Compute cosine similarity between two strings using normalized embeddings."""
    embeddings = model.encode([source, target], normalize_embeddings=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return float(similarity)


def main(text: str, compare: Optional[str], model_path: Path) -> None:
    model = load_model(model_path)

    if compare is not None:
        similarity = compute_similarity(model, text, compare)
        print(f"Cosine similarity: {similarity:.4f}")
    else:
        embedding = model.encode(text, normalize_embeddings=True)
        preview = " ".join(f"{value:.4f}" for value in embedding[:8])
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First values: {preview} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local inference with the RRFSAVANTMADE model.")
    parser.add_argument("text", help="Text to embed or compare against.")
    parser.add_argument(
        "--compare",
        help="Optional second text. When provided, cosine similarity is computed between the two.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent,
        help="Path to the local SentenceTransformer model directory (default: repository root).",
    )

    args = parser.parse_args()
    main(args.text, args.compare, args.model_path)
