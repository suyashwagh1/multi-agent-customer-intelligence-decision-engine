from pathlib import Path

POLICIES_DIR = Path(__file__).parent.parent.parent / "data" / "policies"


def chunk_markdown_file(path: Path):
    """Returns a list of dicts: {doc_title, section_title, content}."""
    text = path.read_text()
    lines = text.split("\n")

    doc_title = path.stem.replace("_", " ").title()
    chunks = []
    current_section = None
    current_lines = []

    def flush():
        if current_section is not None and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append({
                    "doc_title": doc_title,
                    "section_title": current_section,
                    "content": content,
                })

    for line in lines:
        if line.startswith("# ") and not line.startswith("## "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            current_section = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return chunks


def chunk_all_policies():
    """Returns all chunks across every .md file in data/policies/."""
    all_chunks = []
    for path in sorted(POLICIES_DIR.glob("*.md")):
        chunks = chunk_markdown_file(path)
        for c in chunks:
            c["doc_filename"] = path.name
        all_chunks.extend(chunks)
    return all_chunks