"""Splits papers on paragraphs and saves each paragraph as a separate document for retrieval."""

import json


def main():
    with open("data/full_text_db_with_missing_cite_ids_extended.json") as f:
        full_texts = json.load(f)

    for paper_id in full_texts:
        full_text = full_texts[paper_id]
        with open(
            f"data/emnlp23/docs-for-retrieval/{paper_id}.jsonl", "w"
        ) as f:
            for section_i, section in enumerate(
                [
                    {
                        "section_name": full_text["title"],
                        "paragraphs": [full_text["abstract"]],
                    }
                ]
                + full_text["full_text"]
            ):
                for para_i, paragraph in enumerate(section["paragraphs"]):
                    f.write(
                        json.dumps(
                            {
                                "did": f"{paper_id}.s{section_i}p{para_i}",
                                "text": paragraph,
                                "section": section["section_name"],
                            }
                        )
                        + "\n",
                    )


if __name__ == "__main__":
    main()
