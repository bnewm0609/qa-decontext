"""Utilities needed for initalizing datasets in `data.py`.

These are in a separate file from `utils.py` to avoid circular imports as because certain utility functions
in `utils.py` require classes from `data.py`, and `data.py` requires these utilities.
"""

from typing import Optional

from pydantic import BaseModel
from transformers import PreTrainedTokenizer


class OpenAIChatMessage(BaseModel):
    """Input the the OpenAI chat Endpoint
    
    Attributes:
        role (str): "system" or "user".
        content (str): the prompt to the model.
    """

    role: str
    content: str


class QasperSection(BaseModel):
    """Section of a paper in Qasper
    
    Attributes:
        section_name (str): The section heading.
        paragraphs (list[str]): the paragrphs in the section.
    """

    section_name: str
    paragraphs: list[str]


class QasperFullText(BaseModel):
    """The full text of a paper in Qasper
    
    Attributes:
        title (str): The paper title.
        abstract (str): The paper abstract.
        full_text (list[QasperSection]): The sections that constitute the paper.
        word_limit (int): The maximum number of space-separated tokens to include in the string representation
            of the paper. This is included to help avoid exceeding token limits of models. (Though note that
            additional post-processing of full-texts is done (eg in `data.py`) because model tokens and
            space-sparated tokens are not the same.)
    """

    title: str
    abstract: str
    full_text: list[QasperSection]
    word_limit: int = 5_000

    def __str__(self):
        """Format the QasperFullText
        
        The Full Texts are formatted as secion headings paragraphs separated by two new line characters.
        Full texts are truncated at the word limit.
        """

        out_str = ""
        out_str += self.title + "\n\n"
        out_str += self.abstract + "\n\n"

        for section in self.full_text:
            section_name = section.section_name
            section_name = section_name.split(" ::: ")[-1]
            out_str += section_name + "\n\n"
            out_str += "\n".join(section.paragraphs) + "\n\n"

        if self.word_limit > 0:
            out_str = " ".join(out_str.split(" ")[: self.word_limit])

        return out_str
