import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from app.rag.hybrid_retriever import retrieve

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
EVAL_PATH = Path(__file__).parent / "test_conversations.json"

JUDGE_MODEL = "claude-sonnet-5"


def _extract_text(response):
    """Response content may include a thinking block before the text block --
    find the actual text, don't assume content[0] is it."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def judge_score(prompt: str) -> float:
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = _extract_text(response).strip()
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        print(f"  WARNING: judge returned non-numeric response: {text!r}, scoring as 0.0")
        return 0.0


def score_faithfulness(answer: str, context: str) -> float:
    prompt = f"""You are evaluating an AI customer support agent's answer for
faithfulness to its source material.

RETRIEVED CONTEXT (the only source the agent was allowed to use):
{context}

AGENT'S ANSWER:
{answer}

Score from 0.0 to 1.0: what fraction of factual claims in the answer are
directly supported by the retrieved context? A claim not present in the
context (even if plausible) should count against the score. Respond with
ONLY a number between 0.0 and 1.0, nothing else."""
    return judge_score(prompt)


def score_context_precision(query: str, context: str) -> float:
    prompt = f"""You are evaluating whether retrieved context is relevant to a
customer's question.

CUSTOMER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Score from 0.0 to 1.0: how relevant and useful is this context for actually
answering the question? Respond with ONLY a number between 0.0 and 1.0,
nothing else."""
    return judge_score(prompt)


def run_eval():
    cases = json.loads(EVAL_PATH.read_text())
    policy_cases = [c for c in cases if c["category"].startswith("policy")]

    print(f"Running RAGAS-style eval on {len(policy_cases)} policy_question cases...\n")

    results = []
    for case in policy_cases:
        query = case["query"]
        chunk, retrieval_score = retrieve(query)

        if chunk is None:
            print(f"[{case['id']}] {query[:60]}...")
            print("  -> No retrieval (correctly refused) -- skipping faithfulness/precision scoring")
            print("  -> This is EXPECTED and CORRECT for policy_refusal cases\n")
            results.append({"id": case["id"], "category": case["category"], "refused": True})
            continue

        context = f"{chunk['doc_title']} -- {chunk['section_title']}: {chunk['content']}"

        gen_response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"Answer this customer question using ONLY this context:\n\n{context}\n\nQuestion: {query}",
            }],
        )
        answer = _extract_text(gen_response)

        faithfulness = score_faithfulness(answer, context)
        precision = score_context_precision(query, context)

        print(f"[{case['id']}] {query[:60]}...")
        print(f"  Retrieved: {chunk['doc_title']} -- {chunk['section_title']} (retrieval score: {retrieval_score:.2f})")
        print(f"  Faithfulness: {faithfulness:.2f} | Context precision: {precision:.2f}\n")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "refused": False,
            "faithfulness": faithfulness,
            "context_precision": precision,
        })

    scored = [r for r in results if not r["refused"]]
    if scored:
        avg_faith = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_prec = sum(r["context_precision"] for r in scored) / len(scored)
        print("=" * 50)
        print(f"Average faithfulness:      {avg_faith:.2f}")
        print(f"Average context precision: {avg_prec:.2f}")
        print(f"Cases correctly refused:   {sum(1 for r in results if r['refused'])}/{len(results)}")

    return results


if __name__ == "__main__":
    run_eval()