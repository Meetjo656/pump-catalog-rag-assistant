def row_to_text(source_file: str, row: dict) -> str:
    """
    Convert a single CSV row into a clean text chunk for RAG.
    """

    lines = []

    # Header
    lines.append(f"Source: {source_file}")

    # Convert each column into readable text
    for key, value in row.items():
        if value is None or str(value).strip() == "":
            continue
        lines.append(f"{key}: {value}")

    return "\n".join(lines)
