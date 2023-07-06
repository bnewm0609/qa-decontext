#
#  Much of the functionality comes from `pipeline_run.py` but this implementation
#  doesn't have the initial decontextualization step. Instead it puts it at the
#  end with the synthesis step.

import glob
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from contrastive_tldrs.experiment.api import ApiExperimentRunner
from contrastive_tldrs.utils import hash_strs
from hydra import compose  # , initialize
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, ListConfig, OmegaConf
from shadow_scholar.app import pdod

NO_QUESTIONS = {"No questions.", "Impossible."}

FULL_TEXTS = {}
with open("data/full_text_db_with_missing_cite_ids_extended.json") as f:
    FULL_TEXTS = json.load(f)


@dataclass
class PaperSnippet:
    """A snippet from a Paper along with all of the context needed to perform decontextualization.

    Attributes:
        idx: The index of the snippet.
        paper_id: The S2ORC ID (citences) or ARXIV ID (Qaspar) of the paper the snippet comes from.
        title: The title of the paper the snippet comes from.
        abstract: The abstract of the paper the snippet comes from.
        full_text: list of sections of the form
                {"section_name": "<section_name>", "paragraphs": ["<paragraph_1>", "<paragraph_2>", ...]}
            This is usually not used and relies on the paper_id to get the full text later in the pipeline
            if necessary.
        section_header: The header of the section the of the paper the snippet comes from.
        context: The paragraph of the paper the snippet comes from.
        cited_ids: list of the S2ORC IDs for the papers cited in the snippet (Citences, empty for Qaspar).
        questions: A map from question id to question. These questions can be generated or be human-authored
            depending on the experiment.
        additional_paragraphs: Additional paragraphs that contain important context. These might have be specified
            by people during the annotation task or might be retrieved using a retrieval model operating over the
            paper.
        answers: Answers to the questions according to the above title, abstract, section header, context,
            and additional_paragraphs. Again, can be generated or human-written.
        decontextualized_snippet: The final decontextualized snippet which rewrites the snippet to synthesize
            the answers to the questions.
        y_gold: The reference decontextualized snippet.
    """

    idx: str
    paper_id: str
    title: str
    abstract: str = field(repr=False)
    full_text: list[dict[str, Union[list[str], str]]] = field(repr=False)
    section_header: str
    context: str = field(repr=False)
    snippet: str
    cited_ids: list[str]
    # these should be filled in as the pipeline is run
    snippet_surface: str = field(default_factory=str, repr=False)
    questions: dict[str, str] = field(
        default_factory=dict, repr=False
    )  # qid: question
    additional_paragraphs: dict[str, list[str]] = field(
        default_factory=dict, repr=False
    )
    answers: dict[str, str] = field(
        default_factory=dict, repr=False
    )  # qid: answer
    decontextualized_snippet: str = field(default_factory=str)
    y_gold: str = field(default_factory=str)

    def __getitem__(self, item):
        return getattr(self, item)


class PipelineStep:
    """Run a component of the pipeline.

    Instead of launching a new process to run a pipeline component, load in the config,
    and update the globabl config store with that pipeline's config and run as normal.

    Attributes:
        args: The parameters for this step of the experiment.
        experiment_runner: The object that runs the experiments.
    """

    def __init__(self, step_name, conf="configs/config.yaml", overrides=None):
        """Initialize the step of the pipeline.

        Loads in the conf with the overrides

        Args:
            step_name (str): Name of the step (e.g. "qgen", "qa", "synth").
            conf (Path): Path to the config for the step.
            overrides (list[str]): Overrides to the config. These would be passed at the command line if running
                a separate process. Each element of the list is an override with the following format: "key=val".
        """

        try:
            OmegaConf.register_new_resolver(
                "esc_slash", lambda x: x.replace("/", "-")
            )
            OmegaConf.register_new_resolver(
                "esc_period", lambda x: str(x).replace(".", "-")
            )
            # ignores trailing "/"
            OmegaConf.register_new_resolver(
                "extract_path",
                lambda path, num_segments: "-".join(
                    path.strip("/").split("/")[-num_segments:]
                ),
            )
            OmegaConf.register_new_resolver("hash", hash_strs)
        except ValueError:
            # The resolvers are already registered
            print("Resolvers already registered")

        # following here: https://github.com/facebookresearch/hydra/issues/1022
        pipeline_step_config = OmegaConf.load(conf)
        # ConfigStore.instance().store(f"pipeline_step_config", node=pipeline_args)
        ConfigStore.instance().store(
            f"{step_name}_config", node=pipeline_step_config
        )
        self.args = compose(
            config_name=f"{step_name}_config", overrides=overrides
        )
        # print(self.args)
        # with initialize(version_base=None, config_path="../configs"):
        #     self.args = compose(config_name=conf, overrides=overrides)

        self.experiment_runner = ApiExperimentRunner(self.args)

    def run(self):
        """Run the step of the experiment."""

        self.experiment_runner.predict(
            self.args,
            self.experiment_runner.data,
            self.experiment_runner.model,
            "val",
        )

    @staticmethod
    def prepare_overrides(
        step: str, args: DictConfig, results_path: Path, in_data_path: Path
    ) -> list[str]:
        """Create the overrides list for a number of overrides common to all steps."""
        overrides = ["mode=predict"]
        overrides += ["+warn_before_hitting_api=False"]
        overrides += [f"results_path={results_path}"]
        if (
            args.model[step].get("data") is not None
        ):  # should be None except for wiki
            overrides += [
                f"data={args.model[step].data}",
                f"data.val.path={in_data_path}",
            ]
        else:
            overrides += ["data=rewrite", f"data.val.path={in_data_path}"]
        overrides += [f"model={args.model[step].model}"]
        overrides += [f"generation={args.model[step].generation}"]
        # add overrides:
        overrides_extra = args.model[step].get("overrides", {})
        for key, val in overrides_extra.items():
            if isinstance(val, str):
                val = val.replace('"', '\\"')
            elif isinstance(val, ListConfig):
                val = [v["content"].replace('"', '\\"') for v in val]
            overrides.append(f'{key}="{val}"')
        return overrides


def load_steps_to_run(args: DictConfig) -> list[str]:
    """Determine what steps to run based on the given args and return them.

    The args specify which steps have gold data, and that should be used instead of generated data for certain
    ablation experiments.

    Returns:
        A list of the steps that should be run.
    """

    steps = ["qgen", "qa-retrieval", "qa-answers", "synth"]
    use_gold_data = [
        args.model.qgen.get("use_gold", False),
        args.model.qa.get("use_gold_evidence", False),
        args.model.qa.get("use_gold_answers", False),
    ]
    skip_steps = [False] * len(steps)

    # If using gold data for a later step, also use it for the previous steps.
    # Eg. otherwise you will have gold answers to generated questions and they won't match up.
    for i, step_uses_gold_data in enumerate(use_gold_data):
        if step_uses_gold_data:
            skip_steps = ([True] * (i + 1)) + skip_steps[i:]

    return [step for i, step in enumerate(steps) if not skip_steps[i]]


def load_data(data_path: str, run_steps=None) -> list[PaperSnippet]:
    """Load in the data specified in the `PaperSnippet` class.

    If a run step is skipped, then all of the data for that step and previous steps will be gold,
    human-written data. As written, these are pre-extracted, but in the future, they should be extracted from the
    full-text or given a s2orc id.

    Args:
        data_path: path to data
        run_steps: The steps to extract data for. If not provided, extract for all of the steps.

    Returns:
        A list of PaperSnippets derived from the data.
    """
    if run_steps is None:
        run_steps = ["qgen", "qa-retrieval", "qa-answers", "synth"]
    with open(data_path) as f:
        data_instances = [json.loads(line) for line in f]

    snippets = []
    for data in data_instances:
        snippet = PaperSnippet(
            idx=data["idx"],
            paper_id=data["paper_id"],
            title=data["title"],
            abstract=data["abstract"],
            full_text=[{}],  # data["full_text"],
            section_header=data["context_section_header"],
            context=data["context_paragraph"],
            snippet=data["sentence"],
            cited_ids=[e["paper_id"] for e in data["cited_ids"]],
            y_gold=data["y"] if "y" in data else "",
        )

        if "qgen" not in run_steps:
            snippet.questions = data["questions"]
        if "qa-retrieval" not in run_steps:
            snippet.additional_paragraphs = {
                qid: [evidence["paragraph"] for evidence in evidences]
                for qid, evidences in data["evidence"].items()
            }
        if "qa-answers" not in run_steps:
            snippet.answers = data["answers"]

        snippets.append(snippet)
    return snippets


# Here we define the different steps of the pipeline:
def question_generation(args: DictConfig, paper_snippet: PaperSnippet):
    """Run the question generation step of the pipeline and update the paper snippet with the generated questions.

    If the snippets already have questions, skip this step. If the results file already exists, then load in the
    snippets rather than generating them again.

    Args:
        args: Retrieval arguments.
        paper_snippet: The snippet being decontextualized. This is where the generated questions will be placed.

    Returns:
        A string with the Path to where the predictions are stored.
    """

    if paper_snippet.questions:
        return

    def local_print(*args, **kwargs):
        print("[QGen]", *args, **kwargs)

    exp_dir = Path(args.results_path) / "qgen"
    exp_dir.mkdir(exist_ok=True)

    # check if we have already run this to create the predictions
    predictions_path = glob.glob(str(exp_dir / "*/predictions.json"))
    if predictions_path:
        local_print("Experiment directory already exists! Skipping step...")
        local_print(predictions_path[0])
    else:
        # create the data there:
        # TODO: this data will probably take a different format depending
        # on what the prompt looks like (eg template-completion-qgen-science.yaml takes
        # a datset with a "sentence")
        in_data_path = exp_dir / "in.jsonl"
        with in_data_path.open("w") as f:
            # Different templates require different variables or variable names.
            # They will ignore what they don't need, so just dump everything that could be needed here.
            json.dump(
                {
                    "idx": paper_snippet.idx,
                    "x": f'Paper Title: {paper_snippet.title}\nText: "{paper_snippet.snippet}"',
                    "title": paper_snippet.title,
                    "sentence": paper_snippet.snippet,
                    "snippet": paper_snippet.snippet,  # this is intentionally the same as above
                    "y": "",
                },
                f,
            )

        # run decontextualization:
        overrides = PipelineStep.prepare_overrides(
            "qgen", args, exp_dir, in_data_path
        )
        qgen_step = PipelineStep("question_generation", overrides=overrides)
        qgen_step.run()

        predictions_path = glob.glob(str(exp_dir / "*/predictions.json"))

    # read the predictions and update the paper_snippet object
    with open(predictions_path[0]) as f:
        prediction = json.load(f)["y_hat"]

        # splits at line-breaks, strips whitespace and bullets from each line, and removes empty lines
        def my_strip(s):
            return s.lstrip(" -*").strip()

        prediction = list(filter(bool, map(my_strip, prediction.splitlines())))

        for question in prediction:
            qid = f"{paper_snippet.idx}.{hash_strs(question, lim=10)}"
            paper_snippet.questions[qid] = question

    return predictions_path


def run_retrieval(args: DictConfig, paper_snippet: PaperSnippet):
    """Run retrieval and update the passed paper_snippet object with the result.

    The questions are used as queries, and are batched into one `query,.jsonl` file.
    If the retrieval output path already exists, we don't run retrieval again and just use the results from the
    previous run.

    Args:
        args: Retrieval arguments.
        paper_snippet: The snippet being decontextualized. This is where the retrieved data will be placed.
    """

    exp_dir = Path(args.results_path) / "qa"
    exp_dir.mkdir(exist_ok=True)

    def local_print(*args, **kwargs):
        print("[QA-Retrieval]", *args, **kwargs)

    # Collect all of the questions as queries for retrieval
    # if statements prevent duplicating work
    retrieval_output_path = exp_dir / "retrieval_results.jsonl"
    requires_retrieval = False
    if not retrieval_output_path.exists():
        query_path = exp_dir / "query.jsonl"
        if not query_path.exists():
            for qid, question in paper_snippet.questions.items():
                # Check if question requires additional context and run retrieval if it does.
                # For now, assume that it does require retrieval.
                requires_additional_context = not any(
                    [question in no_question for no_question in NO_QUESTIONS]
                )
                requires_retrieval = (
                    requires_retrieval or requires_additional_context
                )

                if requires_additional_context:

                    query = ""
                    # Enables testing if adding the snippet to the query improves retrieval results (it doesn't)
                    if args.model.qa.retriever == "dense-snippet-q":
                        query = f'In the passage: "{paper_snippet.snippet}"\n{question}'
                    else:
                        query = question

                    # Save the queries
                    with open(query_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "qid": qid,
                                    "text": query,
                                }
                            )
                            + "\n",
                        )
        else:
            requires_retrieval = True

        if requires_retrieval:
            # run retrieval

            # if we need to use the cited ids then we run retrieval twice - once for
            # each paper, and then we merge the results
            for paper_id in [paper_snippet.paper_id] + paper_snippet.cited_ids:
                # We've pre-processed the full-texts to be compatible with pdod
                # so assumes the docs already exist at this path: (Can use above code to create them if not)
                # TODO: how would we make this more general?
                doc_path = Path(
                    f"data/emnlp23/docs-for-retrieval/{paper_id}.jsonl"
                )
                if not doc_path.exists():
                    local_print(f"Docs don't exist at: {doc_path}")
                    continue
                    # return this

                # run retrieval
                ranker_kwargs = None
                retriever_name = args.model.qa.retriever
                if args.model.qa.retriever == "dense":
                    ranker_kwargs = {
                        "model_name_or_path": "facebook/contriever"
                    }

                if args.model.qa.retriever == "dense-snippet-q":
                    retriever_name = "dense"

                paper_retrieval_output_path = (
                    str(exp_dir / paper_id) + "_retrieval_results.jsonl"
                )
                pdod.main.run_pdod(
                    retriever_name,
                    ranker_kwargs=ranker_kwargs,
                    docs_path=doc_path,
                    queries_path=query_path,
                    output_path=paper_retrieval_output_path,
                )

            # collect the retrieval results for the snippet (across all paper)
            # into one results file at the retrieval_output_path
            docs = []
            for paper_id in [paper_snippet.paper_id] + paper_snippet.cited_ids:
                paper_retrieval_output_path = (
                    str(exp_dir / paper_id) + "_retrieval_results.jsonl"
                )
                try:
                    with open(paper_retrieval_output_path) as f:
                        docs.extend([json.loads(line.strip()) for line in f])
                except FileNotFoundError:
                    print("Cannot find file:", paper_retrieval_output_path)

            docs = sorted(docs, key=lambda doc: doc["score"], reverse=True)
            with open(retrieval_output_path, "w") as f:
                for doc in docs:
                    f.write(json.dumps(doc) + "\n")

    # collect the retrieval results for all snippets
    docs = []
    try:
        with open(retrieval_output_path) as f:
            docs = [json.loads(line.strip()) for line in f]
    except FileNotFoundError:
        local_print("No retrieval needed")
        return

    # group by docs by qid
    docs_by_qid = defaultdict(list)
    for doc in docs:
        docs_by_qid[doc["qid"]].append(doc["text"])

    # TODO: depending on the strategy, add different docs to `additional_paragraphs`
    # field of `paper_snippet`
    # For now, assume strategy is top-3
    for qid in docs_by_qid:
        paper_snippet.additional_paragraphs[qid] = docs_by_qid[qid][:3]  # type: ignore


def question_answering(args: DictConfig, paper_snippet: PaperSnippet):
    """Run question answering and update the passed paper_snippet object with the result.

    If the output path already exists, we don't run QA again and just use the results from the
    previous run.

    Args:
        args: experiment arguments.
        paper_snippet: The snippet being decontextualized. This is where the answers will be stored.
    """

    def local_print(*args, **kwargs):
        print("[QA]", *args, **kwargs)

    exp_dir = Path(args.results_path) / "qa"
    exp_dir.mkdir(exist_ok=True)

    #  We separated out running retrieval bc we want to be able to substitute in gold stuff
    #  run_retrieval(args, paper_snippet, exp_dir)
    # print(paper_snippet.additional_paragraphs)

    # issue one prompt per question (this is more expensive, but more accurate...)
    no_questions = False
    for qid, question in paper_snippet.questions.items():
        if any([question in no_question for no_question in NO_QUESTIONS]):
            no_questions = True
            break
        exp_dir_q = exp_dir / qid.replace("/", "_")
        exp_dir_q.mkdir(exist_ok=True)

        # check if we have already run this to create the predictions
        predictions_path = glob.glob(str(exp_dir_q / "*/predictions.json"))
        if predictions_path:
            local_print(
                f"Experiment directory for question:{qid} already exists! Skipping step..."
            )
            local_print(predictions_path[0])
        else:
            # creates the data there:
            prompt = f"Title: {paper_snippet.title}\n\n"
            if paper_snippet.abstract:
                prompt += f"Abstract: {paper_snippet.abstract}\n\n"

            # just uses the gold retrieved doc for this question - rely on the qi to get this.
            # They should be ordered, one per question
            unique_additional_paragraphs = []
            if paper_snippet.additional_paragraphs:
                for example in paper_snippet.additional_paragraphs[qid]:
                    if (
                        example == paper_snippet.context
                        or example == paper_snippet.abstract
                    ):
                        continue
                    if example not in unique_additional_paragraphs:
                        unique_additional_paragraphs.append(example)
                    prompt += f"Additional Paragraph: {example}\n\n"
            else:
                print("NO ADDITIONAL PARAGRPHS FOUND")

            if paper_snippet.section_header:
                prompt += f"Section Title: {paper_snippet.section_header}\n"
            prompt += f"Context: {paper_snippet.context}\n"
            prompt += "\n"
            prompt += f"Text Snippet: {paper_snippet.snippet}\n\n"
            prompt += "Question: " + question

            full_text: dict = {}
            if "gpt4" in args.model.qa.model:
                full_text = FULL_TEXTS.get(paper_snippet.paper_id, {})

            in_data_path = exp_dir_q / "in.jsonl"
            with in_data_path.open("w") as f:
                # like in question-generation, there are a number of different fields that the templates
                # might include, so we want to include all of them in case they are needed.
                sample = {
                    "idx": qid,
                    "x": prompt,
                    "title": paper_snippet.title,
                    "abstract": paper_snippet.abstract,
                    "context_paragraph": paper_snippet.context,
                    "question": question,
                    "unique_evidence": [
                        {"paragraph": evidence}
                        for evidence in unique_additional_paragraphs  # paper_snippet.additional_paragraphs[qid]
                    ],
                    "sentence": paper_snippet.snippet,
                    "y": "",
                }
                if full_text:
                    sample["full_text"] = full_text  # type: ignore
                json.dump(sample, f)

            overrides = PipelineStep.prepare_overrides(
                "qa", args, exp_dir_q, in_data_path
            )
            overrides += [f"data.train.path={in_data_path}"]
            surface_step = PipelineStep("question_answer", overrides=overrides)
            surface_step.run()

            predictions_path = glob.glob(str(exp_dir_q / "*/predictions.json"))

        # reads the predictions and update the paper_snippet object
        if no_questions:
            paper_snippet.answers[qid] = "No answer."
            return exp_dir

        with open(predictions_path[0]) as f:
            prediction = json.load(f)["y_hat"]

            # strips whitespace from each line
            prediction = prediction.strip()
            paper_snippet.answers[qid] = prediction

    return exp_dir


def synthesize(args: DictConfig, paper_snippet: PaperSnippet):
    """Rewrite the snippets to incorporate the answers to the questions.

    If the output path already exists, we don't run synthesis again and just use the results from the
    previous run.

    Args:
        args: experiment arguments.
        paper_snippet: The snippet being decontextualized. This is where the final decontextualizations will be
            stored.
    """

    def local_print(*args, **kwargs):
        print("[Synth]", *args, **kwargs)

    exp_dir = Path(args.results_path) / "synth"
    exp_dir.mkdir(exist_ok=True)

    # checks if we have already run this to create the predictions
    predictions_path = glob.glob(str(exp_dir / "*/predictions.json"))
    if predictions_path:
        local_print("Experiment directory already exists! Skipping step...")
        local_print(predictions_path[0])
    else:
        # creates the data there:
        in_data_path = exp_dir / "in.jsonl"

        prompt = f"Snippet: {paper_snippet.snippet}\n"
        for qid, question in paper_snippet.questions.items():
            answer = paper_snippet.answers.get(qid, "No answer.")
            if answer == "No answer.":
                continue
            prompt += f"Question: {question}\n"
            prompt += f"Answer: {answer}\n"

        # There are no answers or no questions, so we don't need to synthesize anything.
        if "Answer:" not in prompt:
            paper_snippet.decontextualized_snippet = paper_snippet.snippet
            return exp_dir

        # Deduplicate the evidence.
        unique_evidence = []
        for es in paper_snippet.additional_paragraphs.values():
            unique_evidence.extend(es)
        unique_evidence = list(set(unique_evidence))

        with in_data_path.open("w") as f:
            json.dump(
                {
                    "idx": paper_snippet.idx,
                    "x": prompt,
                    "y": "",  # no y, but includes it bc the run script assumes it's there for eval reasons.
                    "questions": [
                        {
                            "question": question,
                            "answer_text": answer,
                        }
                        for question, answer in zip(
                            paper_snippet.questions.values(),
                            paper_snippet.answers.values(),
                        )
                    ],
                    "sentence": paper_snippet.snippet,
                    "title": paper_snippet.title,
                    "abstract": paper_snippet.abstract,
                    "context_section_header": paper_snippet.section_header,
                    "context_paragraph": paper_snippet.context,
                    "unique_evidence": unique_evidence,
                },
                f,
            )

        overrides = PipelineStep.prepare_overrides(
            "synth", args, exp_dir, in_data_path
        )
        overrides += [f"data.train.path={in_data_path}"]
        synth_step = PipelineStep("synthesis", overrides=overrides)
        synth_step.run()

        predictions_path = glob.glob(str(exp_dir / "*/predictions.json"))

    # reads the predictions and update the paper_snippet object
    with open(predictions_path[0]) as f:
        prediction = json.load(f)["y_hat"]

        # strips whitespace
        prediction = prediction.strip()
        paper_snippet.decontextualized_snippet = prediction

    return predictions_path


def decontextualize_snippet(snippet: PaperSnippet):
    """
    Assume that this is called by some outside source with a package.
    Somehow, we get a paper snippet
    """
    pass


def decontextualize_text(
    snippet: str, context: dict, questions: Optional[list[str]]
):
    """
    context (dict): an additional context that is used during decontextualization.
        Should include the idx, title, abstract, full_text, section_header, context_paragraph,
        and the ids of the cited paper.
    """
    pass
