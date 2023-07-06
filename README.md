# QA Decontextualization

See `experiments` for description of how to run experiments investigating this method.

## Installation
```bash
pip install decontext
```

Or if you prefer to install locally:
```bash
conda create  -n decontext python=3.9
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

To decontextualize a snippet using some context, you can pass both the snippet and context to the decontextualization function. The decontextualization will be best if it includes certain parts of the paper: especially the title, abstract, and the paragraph surrounding the snippet. If these can't be found, a warning will be raised.
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
    paragraph_with_snippet=context_paragraph_3
    additional_paragraphs=[context_paragraph_1, context_paragraph_2]
)
```

In addition to specifying individual paragraphs, you can also use the entire paper full text as context:
```python
from decontext import PaperContext, Section
context = PaperContext(
    title="Identifying Dogmatism in Social Media: Signals and Models",
    abstract="We explore linguistic and behavioral features of dogmatism in social media and construct statistical models that can identify dogmatic comments. Our model is based on a corpus of Reddit posts, collected across a diverse set of conversational topics and annotated via paid crowdsourcing. We operationalize key aspects of dogmatism described by existing psychology theories (such as over-confidence), finding they have predictive power. We also find evidence for new signals of dogmatism, such as the tendency of dogmatic posts to refrain from signaling cognitive processes. When we use our predictive model to analyze millions of other Reddit posts, we find evidence that suggests dogmatism is a deeper personality trait, present for dogmatic users across many different domains, and that users who engage on dogmatic comments tend to show increases in dogmatic posts themselves.",
    full_text=[Section(name="Introduction", paragraphs=["<paragraph 1>", "<paragraph 2>", ...]), ...],
)

decontext(snippet, context)
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

Finally, the `decontext` function also supports using multiple papers as context:
```python
decontext(snippet, [paper_1_context, paper_2_context])
```

3. Customize decontextualization pipeline

By default we use GPT4 to answer based on the full document, but you can customize the different part of the pipeline by including a config in the call to `decontext`. The pipeline consists of three parts: A question generator (`qgen`), a question-answering system (`qa`), and a synthesizer (`synth`) to rewrite the snippets to include the answers to the questions. Each component of the pipeline can be specified using a `decontext.Config` dataclass. The config can also be a path to a yaml file with the same structure. (The values are the defaults.)
```python
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

decontext(snippet, context, config=config)

4. Debugging
For debugging purposes, it's useful to have access to the intermediate outputs of the pipeline. To show these, set the `return_metadata` argument to `True`. The returned `metadata` is an instance of `decontext.Metadata`:
```python
new_snippet, metadata = decontext(snippet, paper_1, return_metadata=True)

>   Metadata({
        "idx": "<unique identifier for the snippet>" ,
        "snippet": "<original snippet>",
        "context": "<context used in decontextualization>",
        "questions": [
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

## Function Declaration
```python
def decontext(
    snippet: str,
    context: Union[str, list[str], decontext.PaperContext, list[decontext.PaperContext]],
    config: Union[decontext.Config, str, Path] = "configs/default.yaml",
    return_metadata: bool = False
) -> Union[str, Tuple[str, decontext.Metadata]]
"""Decontextualizes the snippet using the given context according to the given config.

Args:
    snippet: The text snippet to decontextualize.
    context: The context to incorporate in the decontextualization. This can be:
        * a string with the context.
        * a list of context strings (each item should be a paragraph).
        * a PaperContext object or list of PaperContext objects
    config: The configuration for the pipeline

Returns:
    string with the decontextualized version of the snippet.

    if `return_metadata = True`, additionally return the intermediate results for each step of the pipeline as described above.
"""

```