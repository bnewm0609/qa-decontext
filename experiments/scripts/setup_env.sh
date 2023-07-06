pip install -e .
yes | pip uninstall torchvision torch
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/cu117
python -m spacy download en_core_web_sm
# pip install ../shadow-scholar/
pip install git+https://github.com/huggingface/peft.git