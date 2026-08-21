import psycopg

from app.db.session import DB_URL
from app.rag.chunker import chunk_all_policies
from app.rag.embeddings import embed_batch


def ingest():
    chunks = chunk_all_policies()
    print(f"Chunked {len(chunks)} sections from data/policies/")

    print("Embedding chunks...")
    contents = [c["content"] for c in chunks]
    embeddings = embed_batch(contents)
    print("Embedding done.")

    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM policy_chunks")
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO policy_chunks (doc_filename, doc_title, section_title, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["doc_filename"],
                        chunk["doc_title"],
                        chunk["section_title"],
                        chunk["content"],
                        str(embedding),
                    ),
                )
        conn.commit()

    print(f"Inserted {len(chunks)} policy chunks into Postgres.")


if __name__ == "__main__":
    ingest()