import json

from decontext import PaperContext, EvidenceParagraph, Config, Metadata

def test_paper_context():
    with open("tests/fixtures/snippet.json") as f:
        snippet = json.load(f)

    context_1 = PaperContext(
        title=snippet["title"],
        abstract=snippet["abstract"],
        paragraph_with_snippet={"section": snippet["context_section_header"], "paragraph": snippet["context_paragraph"]},
        additional_paragraphs=[
            EvidenceParagraph(section=evidence["section"], paragraph=evidence["paragraph"])
            for evidences in snippet["evidence"].values()
            for evidence in evidences]
    )

    context_2 = PaperContext(
        title=snippet["title"],
        abstract=snippet["abstract"],
    )

    assert context_1
    assert context_2

    # test parse_raw
    raw_json = json.dumps({
        "title": snippet["title"],
        "abstract": snippet["abstract"]
    })
    assert PaperContext.parse_raw(raw_json)

def test_configs():
    config = Config(
        qgen = {
            "model_name": "text-davinci-003",
            "max_questions": 3,
            "template": "templates/qgen.yaml",
        },
        qa = {
            "retriever": None, # "dense" for contriever or "tfidf" for BM25
            "model_name": "gpt4",
            "template": "templates/qa.yaml",
        },
        synth = {
            "model_name": "text-davinci-003",
            "template": "templates/synth.yaml",
        }
    )

    assert config

def test_metadata():
    metadatum = Metadata(
        idx="123",
        snippet="This is a test snippet",
        context="This is some test context",
        questions=[{
            "qid": "abc", "question": "What is the meaning of life?", "answer": 42,
            "evidence": [{"paragraph": "The meaning of life, the universe and everything is 42."}]
        }],
        decontextualized_snippet="This sinippet is a test snippet.",
        cost=0
    )
    print(metadatum)
    assert metadatum