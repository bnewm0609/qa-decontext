python src/run.py mode=predict model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.use_gold_answers=True model.qa.retriever=dense
python src/run.py mode=evaluate model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.use_gold_answers=True model.qa.retriever=dense

python src/run.py mode=predict model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.retriever=dense
python src/run.py mode=evaluate model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.retriever=dense

python src/run.py mode=predict model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.retriever=dense
python src/run.py mode=evaluate model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.use_gold_evidence=True model.qa.retriever=dense

python src/run.py mode=predict model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae task=pipeline model.qa.retriever=dense model.qgen.use_gold=True
python src/run.py mode=evaluate model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae task=pipeline model.qa.retriever=dense model.qgen.use_gold=True

python src/run.py mode=predict model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.retriever=dense
python src/run.py mode=evaluate model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_qae/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.retriever=dense

python src/run.py mode=predict model=pipeline-gpt4-qa model.qa.data=template-chatgpt4-full-text-qa-2 data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_questions/ task=pipeline model.qgen.use_gold=True
python src/run.py mode=evaluate model=pipeline-gpt4-qa model.qa.data=template-chatgpt4-full-text-qa-2 data=pipeline data.base_dir=data/emnlp23/science/pipeline_gold_questions/ task=pipeline model.qgen.use_gold=True

# -----------------------------
python src/run.py mode=predict model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline  model.qa.retriever=dense model.qgen.use_gold=True
python src/run.py mode=evaluate model=pipeline_better_qgen data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline  model.qa.retriever=dense model.qgen.use_gold=True

python src/run.py mode=predict model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.retriever=dense
python src/run.py mode=evaluate model=pipeline-gpt4-retrieval-qa data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline generation=gpt3-endtoend model.qgen.use_gold=True model.qa.retriever=dense

python src/run.py mode=predict model=pipeline-gpt4-qa model.qa.data=template-chatgpt4-full-text-qa-2 data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline model.qgen.use_gold=True
python src/run.py mode=evaluate model=pipeline-gpt4-qa model.qa.data=template-chatgpt4-full-text-qa-2 data=pipeline data.base_dir=data/emnlp23/science/pipeline_pred_questions/ task=pipeline model.qgen.use_gold=True