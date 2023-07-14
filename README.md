# QA Decontextualization

See `experiments` for description of how to run experiments investigating this method.

## Installation
```bash
pip install decontext
```

Or if you prefer to install locally:
```bash
conda create -n decontext python=3.9
conda activate decontext
git clone https://github.com/bnewm0609/qa-decontextualization.git
pip install -e .
```

## Quick Start

1. Set your OpenAI API key
```bash
export OPENAI_API_KEY='yourkey'
```

2. Decontextualize

To decontextualize a snippet using some context, you can pass both the snippet and context to the decontextualization function.
Currently, the expected format for the context is __entire full-text papers__.
These include the title, abstract, and the sections of the paper.
The `PaperContext` class, which holds these full-texts is a Pydantic model, so its values can be parsed from `json` strings as shown below.
<!-- The decontextualization will be best if it includes certain parts of the paper: especially the title, abstract, and the paragraph surrounding the snippet. If these can't be found, a warning will be raised.
```python
from decontext import decontext

context_paragraph_1 = "Data collection. Subreddits are sub-communities on Reddit oriented around specific interests or topics, such as technology or politics. Sampling from Reddit as a whole would bias the model towards the most commonly discussed content. But by sampling posts from individual subreddits, we can control the kinds of posts we use to train our model. To collect a diverse training dataset, we have randomly sampled 1000 posts each from the subreddits politics, business, science, and AskReddit, and 1000 additional posts from the Reddit frontpage. All posts in our sample appeared between January 2007 and March 2015, and to control for length effects, contain between 300 and 400 characters. This results in a total training dataset of 5000 posts."

context_paragraph_2 = "We compare the predictions of logistic regression models based on unigram bag-of-words features (BOW), sentiment signals (SENT), the linguistic features from our earlier analyses (LING), and combinations of these features. BOW and SENT provide baselines for the task. We compute BOW features using term frequency-inverse document frequency (TF-IDF) and category-based features by normalizing counts for each category by the number of words in each document. The BOW classifiers are trained with regularization (L2 penalties of 1.5)."

context_paragraph_3 = "We now apply our dogmatism classifier to a larger dataset of posts, examining how dogmatic language shapes the Reddit community. Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic). We then use these dogmatism annotations to address four research questions."

context = "\n\n".join([
    context_paragraph_1,
    context_paragraph_2,
    context_paragraph_3,
])

snippet = "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic)."

decontext(snippet, context)
> "[REF0] apply the BOW+LING [bag-of-words and linguistic features] model trained on the full Reddit dataset [different subreddit representing different topics, such as politics, business, science and other other posts in the Reddit home page] to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic)."
```

You can also specify context using a more structured representation. For these situations, the `decontext.PaperContext` dataclass is helpful.

```python
from decontext import PaperContext, Section

context = PaperContext(
    title="Identifying Dogmatism in Social Media: Signals and Models",
    abstract="We explore linguistic and behavioral features of dogmatism in social media and construct statistical models that can identify dogmatic comments. Our model is based on a corpus of Reddit posts, collected across a diverse set of conversational topics and annotated via paid crowdsourcing. We operationalize key aspects of dogmatism described by existing psychology theories (such as over-confidence), finding they have predictive power. We also find evidence for new signals of dogmatism, such as the tendency of dogmatic posts to refrain from signaling cognitive processes. When we use our predictive model to analyze millions of other Reddit posts, we find evidence that suggests dogmatism is a deeper personality trait, present for dogmatic users across many different domains, and that users who engage on dogmatic comments tend to show increases in dogmatic posts themselves.",
    paragraph_with_snippet=context_paragraph_3,
    additional_paragraphs=[context_paragraph_1, context_paragraph_2]
)
```

In addition to specifying individual paragraphs, you can also use the entire paper full text as context: -->
```python
from decontext import PaperContext, Section
context = PaperContext(
    title="Identifying Dogmatism in Social Media: Signals and Models",
    abstract="We explore linguistic and behavioral features of dogmatism in social media and construct statistical models that can identify dogmatic comments. Our model is based on a corpus of Reddit posts, collected across a diverse set of conversational topics and annotated via paid crowdsourcing. We operationalize key aspects of dogmatism described by existing psychology theories (such as over-confidence), finding they have predictive power. We also find evidence for new signals of dogmatism, such as the tendency of dogmatic posts to refrain from signaling cognitive processes. When we use our predictive model to analyze millions of other Reddit posts, we find evidence that suggests dogmatism is a deeper personality trait, present for dogmatic users across many different domains, and that users who engage on dogmatic comments tend to show increases in dogmatic posts themselves.",
    full_text=[Section(section_name="Introduction", paragraphs=["<paragraph 1>", "<paragraph 2>", ...]), ...],
)

snippet = "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic)."

decontext(snippet=snippet, context=context)
```

Subsection names should be separated from their supersection name with ":::". For example, the subsection "Metrics" of the "Methods" section would have the `section_name: "Methods ::: Metrics"`.

The context can be loaded in using the `PaperContext.parse_raw` method:
```python
PaperContext.parse_raw("""{
    "title": "<title>",
    "abstract": "<abstract>",
    "full_text": [{
        "name" : "<section_title>",
        "paragraphs": ["<paragraph>", ...]
    }, ...]
}""")
```

Additionally, the `decontext` function also supports using multiple papers as context:
```python
decontext(snippet=snippet, context=paper_1_context, additional_context=[paper_2_context])
```
The argument `context` should be the one that contains the snippet. The argument `additional_context` can contain other potentially useful material (e.g. papers that are cited in the snippet).

3. Debugging
For debugging purposes, it's useful to have access to the intermediate outputs of the pipeline. To show these, set the `return_metadata` argument to `True`. The returned metadata is an instance of `decontext.PaperSnippet`, which contains these outputs along with the cost of the run.
```python
new_snippet, metadata = decontext(snippet, paper_1, return_metadata=True)

>   PaperSnippet({
        "idx": "<unique identifier for the snippet>" ,
        "snippet": "<original snippet>",
        "context": "<context used in decontextualization>",
        "question": [
            {
                "qid": "<question_1_id>",
                "question": "<question_1>",
                "answer": "<answer>",
                "evidence": [
                    {"section": "<section_name>", "paragraph": "<paragraph>"},
                    ...
                ]
            },
            ...
        ]
        "decontextualized_snippet": "<snippet after decontextualization>"
        "cost": <cost_of_run_in_USD>
    })
```

## Customizing Pipeline Components
If you want to use your own question generation, question answering, or synthesis models as part of the pipeline, you can incorporate them easily.
Each step of the pipeline takes a `decontext.PaperSnippet` instance. This is the data structure that pipeline operates over. Each step fills in a field of the `PaperSnippet`.
* question generation fills in `PaperSnippet.qae.question` by calling `PaperSnippet.add_question(question)`
* question answering optionally fills in `PaperSnippet.qae.Evidence` through retrieval by calling `PaperSnippet.add_additional_paragraphs()`, and fills in `PaperSnippet.qae.answers` by calling `PaperSnippet.add_answer()`. (`qae` stands for "Question, Answer, Evidence").
* synthesis fills in `PaperSnippet.decontextualized_snippet` by calling `PaperSnippet.add_decontextualized_snippet()`
The custom component must call the relevant function for it's part.

Your custom component should inherit from the `decontext.PipelineStep` class and override the `run` method. The method takes only one argument - the `PaperSnippet` object. See the `decontex/steps/{qgen,qa,synth}.py` for examples. this is fine

Under the hood, the `decontext` method does the following:
```python
# 1. Creates the Pipeline object
pipeline = Pipeline(
    steps=[
        TemplateQGenStep(),
        TemplateQAStep(),
        TemplateSynthStep(),
    ]
)

# 2. Create the PaperSnippet object
ps = PaperSnippet(snippet=snippet, context=context, qae=[])

# 3. Runs each component of the pipeline in order
for step in pipeline.steps:
    step.run(ps)
```

Let's say you define your own Question Generation pipeline using a template that's better suited for your data than the default.
```python
from decontext.model import load_model
from decontext.template import Template
from decontext.step.step import QGenStep

class CustomQGen(QGenStep):
    def __init__(self):
        self.gpt3 = load_model("text-davinci-003")
        self.template = Template("""\
Ask clarifying questions about the sinppet that comes from this:

Title: {{title}}
Abstract: {{abstract}}
Snippet: {{snippet}}
Questions:
-""")

    def run(paper_snippet: PaperSnippet):
        prompt = self.template.fill({
            "title": paper_snippet.context.title,
            "abstract": paper_snippet.context.abstract,
            "snippet": paper_snippet.snippet,
        })
        response = self.gpt3(prompt)
        for question in response:
            paper_snippet.add_question(question=response[0])
```

Then, you can incorporate it into the pipeline and pass the pipeline to the `decontext` function:
```python
from decontext import Pipeline

pipeline = Pipeline(steps=[
    CustomQGen(),
    TemplateQAStep(),
    TemplateSynthStep(),
])

decontext(snippet=snippet, context=context, pipeline=pipeline)
```



## Function Declaration
```python
def decontext(
    snippet: str,
    context: PaperContext, # Union[str, List[str], PaperContext, List[PaperContext]],
    additional_contexts: Optional[List[PaperContext]] = None,
    # config: Union[Config, str, Path] = "configs/default.yaml",
    pipeline: Optional[Pipeline] = None,
    return_metadata: bool = False,
) -> Union[str, Tuple[str, PaperSnippet]]:
    """Decontextualizes the snippet using the given context according to the given config.

    Args:
        snippet: The text snippet to decontextualize.
        context: The context to incorporate into the decontextualization. This context must include the snippet.
        additional_contexts: Additional context to use in the decontextualization (eg papers that are cited in the snippet).
        pipeline: The pipeline to run on the snippet.
        return_metadata: Flag for returning the PaperSnippet object with intermediate outputs. (See below).

    Returns:
        string with the decontextualized version of the snippet.

        if `return_metadata = True`, additionally return the intermediate results for each step of the pipeline
        as described above.
    """
```