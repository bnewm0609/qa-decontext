from decontext.data_types import PaperContext
from decontext.pipeline import Pipeline, decontext
from decontext.step.qa import TemplateRetrievalQAStep, TemplateFullTextQAStep
from decontext.step.qgen import TemplateQGenStep
from decontext.step.synth import TemplateSynthStep

# def test_ExperimentQGenStep():
#     step = DefaultQGenStep()

#     snippet = PaperSnippet(
#         snippet="This is a test snippet.",
#         context=PaperContext(
#             title="This is a title.",
#             abstract="This is an abstract.",
#             paragraph_with_snippet=EvidenceParagraph(
#                 section="Section Header",
#                 paragraph="This is a test snippet. This is the next sentence.",
#             ),
#         ),
#         qae=[],
#     )
#     step.run(snippet)

#     assert snippet.qae
#     for question in snippet.qae:
#         print(question.qid, question.question)


# def test_decontext():
#     snippet = (
#         "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
#         "unannotated posts, labeling these posts with a probability of dogmatism according to the"
#         " classifier (0=non-dogmatic, 1=dogmatic)."
#     )

#     context_para_1 = (
#         "Data collection. Subreddits are sub-communities on Reddit oriented around specific "
#         "interests or topics, such as technology or politics. Sampling from Reddit as a whole"
#         " would bias the model towards the most commonly discussed content. But by sampling"
#         " posts from individual subreddits, we can control the kinds of posts we use to train"
#         " our model. To collect a diverse training dataset, we have randomly sampled 1000 posts"
#         " each from the subreddits politics, business, science, and AskReddit, and 1000"
#         " additional posts from the Reddit frontpage. All posts in our sample appeared between"
#         " January 2007 and March 2015, and to control for length effects, contain between 300"
#         " and 400 characters. This results in a total training dataset of 5000 posts."
#     )
#     context_para_2 = (
#         "We compare the predictions of logistic regression models based on unigram bag-of-words"
#         " features (BOW), sentiment signals (SENT), the linguistic features from our earlier"
#         " analyses (LING), and combinations of these features. BOW and SENT provide baselines"
#         " for the task. We compute BOW features using term frequency-inverse document frequency"
#         " (TF-IDF) and category-based features by normalizing counts for each category by the"
#         " number of words in each document. The BOW classifiers are trained with regularization"
#         " (L2 penalties of 1.5)."
#     )
#     context_para_3 = (
#         "We now apply our dogmatism classifier to a larger dataset of posts, examining how dogmatic"
#         " language shapes the Reddit community. Concretely, we apply the BOW+LING model trained"
#         " on the full Reddit dataset to millions of new unannotated posts, labeling these posts"
#         " with a probability of dogmatism according to the classifier (0=non-dogmatic, 1="
#         "dogmatic). We then use these dogmatism annotations to address four research questions."
#     )

#     context = PaperContext(
#         title="Identifying Dogmatism in Social Media: Signals and Models",
#         abstract="We explore linguistic and behavioral features of dogmatism in social media and construct"
#         " statistical models that can identify dogmatic comments. Our model is based on a corpus of "
#         "Reddit posts, collected across a diverse set of conversational topics and annotated via "
#         "paid crowdsourcing. We operationalize key aspects of dogmatism described by existing"
#         " psychology theories (such as over-confidence), finding they have predictive power. We also"
#         " find evidence for new signals of dogmatism, such as the tendency of dogmatic posts to"
#         " refrain from signaling cognitive processes. When we use our predictive model to analyze"
#         " millions of other Reddit posts, we find evidence that suggests dogmatism is a deeper "
#         "personality trait, present for dogmatic users across many different domains, and that users"
#         " who engage on dogmatic comments tend to show increases in dogmatic posts themselves.",
#         paragraph_with_snippet=EvidenceParagraph(paragraph=context_para_3),
#         additional_paragraphs=[
#             EvidenceParagraph(paragraph=context_para_1),
#             EvidenceParagraph(paragraph=context_para_2),
#         ],
#     )

#     paper_snippet = decontext(snippet, context)

#     print(paper_snippet.decontextualized_snippet)


# def test_decontext_retrieval():
#     snippet = (
#         "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
#         "unannotated posts, labeling these posts with a probability of dogmatism according to the"
#         " classifier (0=non-dogmatic, 1=dogmatic)."
#     )

#     with open("tests/fixtures/full_text.json") as f:
#         full_text_json_str = f.read()

#     context = PaperContext.parse_raw(
#         full_text_json_str
#     )  # parse_obj_as(PaperContext, full_text_dict)
#     # add the paragraph with the snippet bc that's important:
#     paragraph_with_snippet = None
#     for section in context.full_text:
#         for paragraph in section.paragraphs:
#             if snippet in paragraph:
#                 paragraph_with_snippet = EvidenceParagraph(
#                     section=section.section_name, paragraph=paragraph
#                 )
#                 break
#         if paragraph_with_snippet is not None:
#             break
#     context.paragraph_with_snippet = paragraph_with_snippet

#     paper_snippet = decontext(snippet, context)

#     print(paper_snippet.decontextualized_snippet)


def test_decontext_template_full_text():
    snippet = (
        "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
        "unannotated posts, labeling these posts with a probability of dogmatism according to the"
        " classifier (0=non-dogmatic, 1=dogmatic)."
    )

    with open("tests/fixtures/full_text.json") as f:
        full_text_json_str = f.read()

    context = PaperContext.parse_raw(full_text_json_str)

    pipeline = Pipeline(
        steps=[
            TemplateQGenStep(),
            TemplateFullTextQAStep(),
            TemplateSynthStep(),
        ]
    )

    decontextualized_snippet = decontext(snippet, context, pipeline=pipeline)
    print(decontextualized_snippet)


def test_decontext_template_retrieval():
    snippet = (
        "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
        "unannotated posts, labeling these posts with a probability of dogmatism according to the"
        " classifier (0=non-dogmatic, 1=dogmatic)."
    )

    with open("tests/fixtures/full_text.json") as f:
        full_text_json_str = f.read()

    context = PaperContext.parse_raw(full_text_json_str)

    pipeline = Pipeline(
        steps=[
            TemplateQGenStep(),
            TemplateRetrievalQAStep(),
            TemplateSynthStep(),
        ]
    )

    decontext_snippet, metadata = decontext(
        snippet, context, pipeline=pipeline, return_metadata=True
    )
    print(metadata.decontextualized_snippet)
    print(metadata.cost)


def test_decontext_template_retrieval_two_contexts():
    snippet = (
        "Concretely, we apply the BOW+LING model trained on the full Reddit dataset to millions of new "
        "unannotated posts, labeling these posts with a probability of dogmatism according to the"
        " classifier (0=non-dogmatic, 1=dogmatic)."
    )

    with open("tests/fixtures/full_text.json") as f:
        full_text_json_str = f.read()

    context_1 = PaperContext.parse_raw(full_text_json_str)

    with open("tests/fixtures/full_text_2.json") as f:
        full_text_2_json_str = f.read()

    context_2 = PaperContext.parse_raw(full_text_2_json_str)

    pipeline = Pipeline(
        steps=[
            TemplateQGenStep(),
            TemplateRetrievalQAStep(),
            TemplateSynthStep(),
        ]
    )

    decontext_snippet, metadata = decontext(
        snippet,
        context_1,
        additional_contexts=[context_2],
        pipeline=pipeline,
        return_metadata=True,
    )
    print(metadata.decontextualized_snippet)
    print(metadata.cost)
