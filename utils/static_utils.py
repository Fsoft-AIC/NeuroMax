import numpy as np
import logging


def print_topic_words(beta, vocab, num_top_words):
    logger = logging.getLogger('main')
    topic_str_list = list()
    for i, topic_dist in enumerate(beta):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(num_top_words + 1):-1]
        topic_str = ' '.join(topic_words)
        topic_str_list.append(topic_str)
        print('Topic {}: {}'.format(i, topic_str))
        logger.info('Topic {}: {}'.format(i, topic_str))
    return topic_str_list
