from pathlib import Path
from typing import List, Optional, Union

from pydantic import BaseModel


# Representing Paper Snippets
class EvidenceParagraph(BaseModel):
    section: Optional[str]
    paragraph: str
    paper_id: Optional[str]


class QuestionAnswerEvidence(BaseModel):
    """Representation of the questions, answers, and evidence."""

    qid: str
    question: str
    answer: Optional[str]
    evidence: Optional[List[EvidenceParagraph]]


# Representing context
class Section(BaseModel):
    """Section of a paper.

    Attributes:
        name (str): The name of the section (ie the section heading).
        paragraphs (list[str]): the paragraphs in the section.
    """

    section_name: str
    paragraphs: List[str]


class PaperContext(BaseModel):
    """The full text of a paper.

    Attributes:
        title (str): The paper title.
        abstract (str): The paper abstract.
        full_text (list[QasperSection]): The sections that constitute the paper.
    """

    title: str
    abstract: str
    full_text: Optional[List[Section]]
    paragraph_with_snippet: Optional[EvidenceParagraph]
    additional_paragraphs: Optional[List[EvidenceParagraph]]

    # class Config:
    #     def from_json(json_str: str):
    #         data = json.loads(str)

    #         if "paragraph_with_snippet" not in data and "full_text" in data:
    #             # parse out the paragraph with the snippet
    #             pass

    #         return data

    #     json_loads = from_json

    def __str__(self):
        """Format the Full Text

        The Full Texts are formatted as secion headings paragraphs separated by two new line characters.
        """

        out_str = ""
        out_str += self.title + "\n\n"
        out_str += self.abstract + "\n\n"

        if self.full_text:
            for section in self.full_text:
                section_name = section.section_name
                section_name = section_name.split(" ::: ")[-1]
                out_str += section_name + "\n\n"
                out_str += "\n".join(section.paragraphs) + "\n\n"

        return out_str.strip()


# Configs
class BaseConfig(BaseModel):
    """All configs have model names and templates"""

    model_name: str
    template: Union[Path, str]


class QGenConfig(BaseConfig):
    """Config for the question generation pipeline component."""

    max_questions: int


class QAConfig(BaseConfig):
    """Config for the question answering pipeline component."""

    retriever: Optional[str]


class SynthConfig(BaseConfig):
    """Config for the question answering pipeline component."""

    pass


class Config(BaseModel):
    qgen: QGenConfig
    qa: QAConfig
    synth: SynthConfig


# Metadata


class Metadata(BaseModel):
    """A snippet from a Paper along with all of the context needed to perform decontextualization.

    Attributes:
        idx: A unique identifier for the snippet.
        snippet: The snippet that is being decontextualized.
        context: The context that was fed into the pipeline.
        questions: A list of questions with their answers and evidences that were created while the pipeline was
            running.
        decontextualized_snippet: The final decontextualized snippet which rewrites the snippet to synthesize
            the answers to the questions.
    """

    idx: str
    snippet: str
    context: Union[str, List[str], PaperContext, List[PaperContext]]
    # these should be filled in as the pipeline is run
    questions: List[QuestionAnswerEvidence]  # qid: question
    decontextualized_snippet: str
    cost: float
