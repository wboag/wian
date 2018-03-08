# notevec
AMIA submission examining note-derived vector representations


Requirements

    1. Python 2.7

    2. postgres MIMIC database


Setup

    $ bash resources/get_resources.sh

    $ pip install -r requirements.txt


Experiments

    $ python code/build_corpus.py all

    $ python code/train_bow.py all

    $ python code/train_embeddings.py all

    $ python code/train_lstm.py all
