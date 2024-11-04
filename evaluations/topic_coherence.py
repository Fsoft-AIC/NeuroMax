from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import numpy as np
from tqdm import tqdm
from itertools import combinations
from datasethandler.file_utils import split_text_word
import os


def compute_topic_coherence(reference_corpus, vocab, top_words, cv_type='c_v'):
    split_top_words = split_text_word(top_words)
    num_top_words = len(split_top_words[0])
    for item in split_top_words:
        assert num_top_words == len(item)

    split_reference_corpus = split_text_word(reference_corpus)
    dictionary = Dictionary(split_text_word(vocab))

    cm = CoherenceModel(texts=split_reference_corpus, dictionary=dictionary,
                        topics=split_top_words, topn=num_top_words, coherence=cv_type)
    cv_per_topic = cm.get_coherence_per_topic()
    score = np.mean(cv_per_topic)

    return cv_per_topic, score


def TC_on_wikipedia(top_word_path, cv_type='C_V'):
    """
    Compute the TC score on the Wikipedia dataset
    """
    jar_dir = "evaluations"
    wiki_dir = os.path.join(".", 'datasets')
    random_number = np.random.randint(100000)
    os.system(
        f"java -jar {os.path.join(jar_dir, 'pametto.jar')} {os.path.join(wiki_dir, 'wikipedia', 'wikipedia_bd')} {cv_type} {top_word_path} > tmp{random_number}.txt")
    cv_score = []
    with open(f"tmp{random_number}.txt", "r") as f:
        for line in f.readlines():
            if not line.startswith("202"):
                cv_score.append(float(line.strip().split()[1]))
    os.remove(f"tmp{random_number}.txt")
    return cv_score, sum(cv_score) / len(cv_score)