# Claude
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp0AQE/UVikc/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp0Vqt9jo+1H/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpTwSRgJ8QkG/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

# - - - 
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpifHedUoGmT/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp2YfcxhzMD6/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp/4hkIkbS4Z/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=claude generation=gpt3-endtoend

# - - - 
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpGLO3fExRKX/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpaM4nYEJVkW/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpp9S6meUhvG/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_answer_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

# - - - 
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp+gjlUJFHpo/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oaclaude-v1_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpLve7bfqRg5/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_claude.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/  model=claude generation=gpt3-endtoend


# GPT-3 Davinci
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmp/xBOo5Q6m9/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpYBQLfiu7wN/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpJmuXRHuGSn/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend


# - - -
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpHU3/zFPoV8/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpil6DIwunc0/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpCDHdsD00sf/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend


# - - -
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpF/PjSyfNYs/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpI9f1YjRCpB/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_answer_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmprD5wWNuYWj/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_gold_context_user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend


# - - -
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpPL5NdckVy7/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_no_title_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/oatext-davinci-003_s1_ep1_lr0-05-trn-science-endtoend_ablations-tmpmh8zU9hdgY/val-science-endtoend_ablations-t-0.7_topp-1.0_mgl-150/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_no_title_completion.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=openai_completion model.name=text-davinci-003 generation=gpt3-endtoend


# Tülu
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpnWKFvLRajh/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpStHsLxnGfK/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpZISmI4xZov/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/tasp_user_intent_answer_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1


# - - -
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpVX/F7Sf0vp/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpGPpJ9G0PIb/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpLYFXS8EXtm/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_context_user_intent_answer_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1


# - - -
python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmp5tIUnhYe2q/val-science-endtoend_ablations-nb-1_mgl-1024/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/user_intent_tulu.yaml  data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=1 generation=rewrite model.precision=16 generation.max_gen_length=1024 generation.num_beams=1

python decontext_exp/run.py mode=predict task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_lama.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=4 generation=rewrite model.precision=16 generation.max_gen_length=512 generation.num_beams=1
python scripts/add_original_sentence_to_metadata.py data/emnlp/all_data.jsonl results/decontext/endtoend/mn-big_models-finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi-_sd-780_bs-1_ga-32_p16_ep-90_lr-3e-05_wu-0.1-trn-science-endtoend_ablations-tmpXL79J/rUF3/val-science-endtoend_ablations-nb-1_mgl-512/metadata.jsonl
python decontext_exp/run.py mode=evaluate task=decontext/endtoend data=template-endtoend-tsp data.template=configs/templates/gold_answer_user_intent_lama.yaml data.base_dir=data/emnlp23/science/endtoend_ablations/ model=tulu model.name=big_models/finetuned_30B_flanv2_cot_oasst1_dolly_sharegpt_gpt4alpaca_codealpaca_lumi/ model.eval_batch_size=4 generation=rewrite model.precision=16 generation.max_gen_length=512 generation.num_beams=1
