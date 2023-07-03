# QA Decontextualization

See `experiments` for description of how to run experiments investigating this method.

## Set Up

```bash
conda create  -n qa_decontext python=3.9
conda activate qa_decontext
pip install -e .
```

## Quick Start

1. Set your OpenAI API key
```bash
export OPENAI_API_KEY='yourkey'
```

2. Decontextualize

To decontextualize a snippet using some context, you can pass both the snippet and context to the decontextualization function.
```python
from qa_decontext import decontextualize

context = """\
Data collection. Subreddits are sub-communities on Reddit oriented around specific interests or topics, such as technology or politics. Sampling from Reddit as a whole would bias the model towards the most commonly discussed content. But by sampling posts from individual subreddits, we can control the kinds of posts we use to train our model. To collect a diverse training dataset, we have randomly s
ampled 1000 posts each from the subreddits politics, business, science, and AskReddit, and 1000 additional posts from the Reddit frontpage. All posts in our sample appeared between January 2007 and March 2015, and to control for length effects, contain between 300 and 400 characters. This results in a total training dataset of 5000 posts.

We compare the predictions of logistic regression models based on unigram bag-of-words features (BOW), sentiment signals (SENT), the linguistic features from our earlier analyses (LING), and combinations of these features. BOW and SENT provide baselines for the task. We compute BOW features using term frequency-inverse document frequency (TF-IDF) and category-based features by normalizing counts for each category by the number of words in each document. The BOW classifiers are trained with regularization (L2 penalties of 1.5).

We now apply our dogmatism classifier to a larger dataset of posts, examining how dogmatic language shapes the Reddit community. Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic). We then use these dogmatism annotations to address four research questions."""

snippet = "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic)."

decontextualize(snippet, context)
> "[REF0] apply the BOW+LING [bag-of-words and linguistic features] model trained on the full Reddit dataset [different subreddit representing different topics, such as politics, business, science and other other posts in the Reddit home page] to millions of new unannotated posts, labeling these posts with a probability of dogmatism according to the classifier (0=non-dogmatic, 1=dogmatic)."
```

Often times, you want to use a whole paper or a list of papers as context. You can specify the papers as paths to `json` files with a specific format or as S2ORC IDs.
```python
paper_1 = "path/to/paper.json"
paper_2 = "path/to/cited_paper.json"

decontextualize(snippet, paper_1)
decontextualize(snippet, [paper_1, paper_2])
```

Below is the `json` file format:
```json
{
    "title": "<title>",
    "abstract": "<abstract>",
    "full_text": [{
        "section_name" : "<section_title>",
        "paragraphs": ["<paragraph>", ...]
    }, ...]
}
```
Subsection names should be separated from their supersection name with ":::". For example, the subsection "Metrics" of the "Methods" section would have the `section_name: "Methods ::: Metrics"`.

You can also specify papers with S2ORC IDs if the full text of the paper as long as the full text of the paper is in S2ORC.
```python
decontextualize(snippet, "52118895")
decontextualize(snippet, ["52118895", "65318895"])
```

3. Customize decontextualization pipeline

By default we use GPT4 to answer based on the full document, but you can customize the different part of the pipeline by including a config in the call to `decontextualize`. The pipeline consists of three parts: A question generator (`qgen`), a question-answering system (`qa`), and a synthesizer (`synth`) to rewrite the snippets to include the answers to the questions. Each component of the pipeline can be specified using the a config. The config can either be a dictionary or a path to a yaml file with the following structure. (The values are the defaults.)
```yaml
qgen:
    model_name: "text-davinci-003"
    max_questions: 3
    template: "templates/qgen.yaml"
qa:
    retriever: null  # "dense" for contriever or "tfidf" for BM25
    model_name: "gpt4"
    template: "templates/qa.yaml"
synth:
    model_name: "text-davinci-003"
    template: "templates/synth.yaml"
```

```python
decontextualize(snippet, context, config="config.yaml")
```

4. Debugging
For debugging purposes, it's useful to have access to the intermediate outputs of the pipeline. To show these, set the `return_metadata` argument to `True`.
```python
new_snippet, metadata = decontextualize(snippet, paper_1, return_metadata=True)
```

The returned `metadata` has the following structure:
```json
{
    "qgen": [
        {"qid": "<question_1_id>", "question": "<question_1>"},
        ...
    ],
    "qa-retrieval": [
        {"<question_1_id>": ["<doc_1>", "<doc_2>", ...]},
        ...
    ],
    "qa-answers": {
        "<question_1_id>": "answer",
        "<question_2_id>": "<answer>",
        ...
    },
    "cost": <cost_in_dollars>
}
```

## Documentation
```python
def decontextualize(
    snippet: str,
    context: Union[str, list[str], Path, list[Path]],
    config: Union[dict, str, Path] = "configs/default.yaml",
    return_metadata: bool = False
) -> Union[str, Tuple[str, dict]]
"""Decontextualizes the snippet using the given context according to the given config.

Args:
    snippet: The text snippet to decontextualize.
    context: The context to incorporate in the decontextualization. This can be:
        * a string with the context.
        * a list of context strings (each item should be a paragraph).
        * a path to a json file with the full paper.
        * a list of paths to json files with the full paper, all of which should be used as context.
        * a string containing the S2ORC ID of a paper to use.
        * a list of strings containing the S2ORC IDs of the papers to use as context.

Returns:
    string with the decontextualized version of the snippet.

    if `return_metadata = True`, additionally return the intermediate results for each step of the pipeline as described above.
"""

```