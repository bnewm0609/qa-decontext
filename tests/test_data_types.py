import json

from decontext import PaperContext, EvidenceParagraph


def test_paper_context():
    with open("tests/fixtures/snippet.json") as f:
        snippet = json.load(f)

    context_1 = PaperContext(
        title=snippet["title"],
        abstract=snippet["abstract"],
        paragraph_with_snippet={
            "section": snippet["context_section_header"],
            "paragraph": snippet["context_paragraph"],
        },
        additional_paragraphs=[
            EvidenceParagraph(
                section=evidence["section"], paragraph=evidence["paragraph"]
            )
            for evidences in snippet["evidence"].values()
            for evidence in evidences
        ],
    )

    context_2 = PaperContext(
        title=snippet["title"],
        abstract=snippet["abstract"],
    )

    assert context_1
    assert context_2

    # test parse_raw
    raw_json = json.dumps(
        {"title": snippet["title"], "abstract": snippet["abstract"]}
    )
    assert PaperContext.parse_raw(raw_json)


def test_paper_context_full_text():
    snippet = (
        "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
        "unannotated posts, labeling these posts with a probability of dogmatism according to the"
        " classifier (0=non-dogmatic, 1=dogmatic)."
    )

    with open("tests/fixtures/full_text.json") as f:
        full_text_json = json.load(f)

    ps = PaperContext.from_full_text_dict(
        full_text_dict=full_text_json, snippet=snippet
    )
    assert ps
