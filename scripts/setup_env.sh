pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
tar -xvf data/full_texts.tar.gz
tar -xvf data/emnlp23/full_texts.tar.gz
# python -m spacy download en_core_web_sm