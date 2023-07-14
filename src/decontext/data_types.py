from typing import List, Optional, Dict

from pydantic import BaseModel, validator

from decontext.utils import hash_strs


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
    additional_paragraphs: Optional[List[EvidenceParagraph]]
    id: str = ""

    @validator("id", pre=True, always=True)
    def set_default_id(cls, v, values):
        if not v:
            return hash_strs(
                [values["title"], values["abstract"], str(values["full_text"])]
            )
        return v

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


class PaperSnippet(BaseModel):
    """Holds all information about a snippet as it is processed by the pipeline.

    This includes the PaperContext, the qae (questions, answers, and evidence), the snippet itself, the final
    decontextualized version of the snippet, and the cost of decontextualizing the snippet (not taking caching
    into account)
    """

    id: str = "0"
    snippet: str
    context: PaperContext
    additional_contexts: Optional[List[PaperContext]] = None
    qae: List[QuestionAnswerEvidence]
    decontextualized_snippet: Optional[str]
    cost: float = 0
    paragraph_with_snippet: EvidenceParagraph = None  # type: ignore

    @validator("paragraph_with_snippet", pre=True, always=True)
    def extract_paragraph_with_snippet(cls, v, values):
        # if there's a value for paragraph_with_snippet, return it
        if v:
            return v

        # extract the paragraph that the snippet is in from the paper
        paragraph_with_snippet = None
        for section in values["context"].full_text:
            for paragraph in section.paragraphs:
                if values["snippet"] in paragraph:
                    paragraph_with_snippet = EvidenceParagraph(
                        section=section.section_name, paragraph=paragraph
                    )
                    break
            if paragraph_with_snippet is not None:
                break

        if paragraph_with_snippet is None:
            raise ValueError(
                "Could not find snippet in the full text! Please make sure the snippet is there."
            )

        return paragraph_with_snippet

    def add_question(self, question: str, qid: Optional[str] = None):
        if qid is None:
            qid = hash_strs([question])
        self.qae.append(
            QuestionAnswerEvidence(
                qid=qid,
                question=question,
            )
        )

    def add_evidence_paragraphs(
        self,
        qid: str,
        additional_paragraphs: List[str],
        sections: Optional[List[str]] = None,
        paper_id: Optional[str] = None,
    ):
        if sections is None:
            sections = [""] * len(additional_paragraphs)

        for qae in self.qae:
            if qae.qid == qid:
                if qae.evidence is None:
                    qae.evidence = []
                for section, additional_paragraph in zip(
                    additional_paragraphs, sections
                ):
                    qae.evidence.append(
                        EvidenceParagraph(
                            section=section,
                            paragraph=additional_paragraph,
                            paper_id=paper_id,
                        )
                    )

    def add_answer(self, qid: str, answer: str):
        for qae in self.qae:
            if qae.qid == qid:
                qae.answer = answer

    def add_decontextualized_snippet(self, decontextualized_snippet):
        self.decontextualized_snippet = decontextualized_snippet

    def add_cost(self, cost):
        self.cost += cost


# Modeling
class ModelResponse(BaseModel):
    cost: Optional[float]


class OpenAIChatMessage(BaseModel):
    """Input the the OpenAI chat Endpoint

    Attributes:
        role (str): "system" or "user".
        content (str): the prompt to the model.
    """

    role: str
    content: str


class OpenAILogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]
    top_logprobs: List[Dict[str, float]]
    text_offset: List[int]


class OpenAICompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[OpenAILogProbs]
    finish_reason: str


class OpenAIChatChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: str


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(ModelResponse):
    id: str
    object: str
    created: int
    usage: OpenAIUsage

    class Config:
        arbitrary_types_allowed = True


class OpenAICompletionResponse(OpenAIResponse):
    """Output of the OpenAI Completion API"""

    model: str
    choices: List[OpenAICompletionChoice]
    usage: OpenAIUsage


class OpenAIChatResponse(OpenAIResponse):
    """Output of the OpenAI Chat API"""

    choices: List[OpenAIChatChoice]


class AnthropicResponse(ModelResponse):
    """Output of the Anthropic API"""

    completion: str
    stop_reason: str
    model: str
