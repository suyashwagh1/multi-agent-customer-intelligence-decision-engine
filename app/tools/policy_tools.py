from langchain_core.tools import tool

from app.rag.hybrid_retriever import retrieve


@tool
def retrieve_policy(query: str) -> str:
    """Search company policy documents for the answer to a policy question.

    Use this for any question about refund eligibility, return windows,
    shipping SLAs, retention discount rules, or escalation criteria.
    Returns the most relevant policy section with its source, or
    indicates no confident match was found -- in that case, do not
    answer from your own knowledge, escalate instead.
    """
    chunk, score = retrieve(query)

    if chunk is None:
        return (
            "NO_CONFIDENT_MATCH: No policy section was found with sufficient "
            "confidence to answer this question. Do not guess -- escalate this "
            "to a human agent instead."
        )

    return (
        f"SOURCE: {chunk['doc_title']} -- {chunk['section_title']}\n"
        f"CONTENT: {chunk['content']}\n"
        f"(retrieval confidence score: {score:.2f})"
    )