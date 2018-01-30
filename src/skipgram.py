from __future__ import division  # py3 "true division"

import logging
import sys
import os
import heapq
import copy
import numpy as np

from timeit import default_timer
from copy import deepcopy
from collections import defaultdict
import threading
import itertools

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab
from six import iteritems, itervalues, string_types
from six.moves import xrange
from types import GeneratorType
import random

logger = logging.getLogger(__name__)

try:
    from gensim.models.word2vec_inner import train_batch_sg, train_batch_cbow
    from gensim.models.word2vec_inner import score_sentence_sg, score_sentence_cbow
    from gensim.models.word2vec_inner import FAST_VERSION, MAX_WORDS_IN_BATCH
except ImportError:
    # failed... fall back to plain numpy (20-80x slower training than the above)
    FAST_VERSION = -1
    MAX_WORDS_IN_BATCH = 10000

# modified hierarchical softmax model based on Gensim's implementation
class Word2Vec_hs_loss(Word2Vec):
    def __init__(self, sentences=None, **kwargs):
        self.inner_node_index_map = {}
        kwargs["hs"] = 1
        kwargs["alpha"] = kwargs.get("alpha", 0.025)
        kwargs["min_alpha"] = kwargs.get("min_alpha", 0.001)
        kwargs["min_count"] = 0
        kwargs["negative"] = 0
        kwargs["sample"] = kwargs.get("sample", 1e-3)
        kwargs["workers"] = kwargs.get("workers", 20)
        super(self.__class__, self).__init__(sentences, **kwargs)

    # add a word as the child of current word in the coarser graph
    def add_word(self, word, parent_word, emb, cur_index):
        fake_vocab_size = int(1e7)
        word_index = len(self.vocab)
        inner_node_index = word_index - 1
        parent_index = self.vocab[parent_word].index

        # add in the left subtree
        if word != parent_word:
            self.vocab[word] = Vocab(index=word_index, count=fake_vocab_size-word_index,sample_int=(2**32))
            if emb is not None:
                self.syn0[cur_index] = emb
            else:
                self.syn0[cur_index] = self.syn0[parent_index]
            # the node in the coarsened graph serves as an inner node now
            self.index2word.append(word)
            self.vocab[word].code = array(list(self.vocab[parent_word].code) + [0], dtype=uint8)
            self.vocab[word].point = array(list(self.vocab[parent_word].point) + [inner_node_index], dtype=uint32)
            self.inner_node_index_map[parent_word] = inner_node_index
        else:
            if emb is not None:
                self.syn0[parent_index] = emb
            self.vocab[word].code = array(list(self.vocab[word].code) + [1], dtype=uint8)
            self.vocab[word].point = array(list(self.vocab[word].point) + [self.inner_node_index_map[word]], dtype=uint32)

    def train(self, sentences, total_words=None, word_count=0,
             total_examples=None, queue_factor=2, report_delay=0.1):
        """
        Update the model's neural weights from a sequence of sentences (can be a once-only generator stream).
        For Word2Vec, each sentence must be a list of unicode strings. (Subclasses may accept other examples.)
        To support linear learning-rate decay from (initial) alpha to min_alpha, either total_examples
        (count of sentences) or total_words (count of raw words in sentences) should be provided, unless the
        sentences are the same as those that were used to initially build the vocabulary.
        """
        self.loss = {}
        if FAST_VERSION < 0:
            import warnings
            warnings.warn("C extension not loaded for Word2Vec, training will be slow. "
                          "Install a C compiler and reinstall gensim for fast training.")
            self.neg_labels = []
            if self.negative > 0:
                # precompute negative labels optimization for pure-python training
                self.neg_labels = zeros(self.negative + 1)
                self.neg_labels[0] = 1.

        logger.info(
            "training model with %i workers on %i vocabulary and %i features, "
            "using sg=%s hs=%s sample=%s negative=%s",
            self.workers, len(self.vocab), self.layer1_size, self.sg,
            self.hs, self.sample, self.negative)

        if not self.vocab:
            raise RuntimeError("you must first build vocabulary before training the model")
        if not hasattr(self, 'syn0'):
            raise RuntimeError("you must first finalize vocabulary before training the model")

        if total_words is None and total_examples is None:
            if self.corpus_count:
                total_examples = self.corpus_count
                logger.info("expecting %i sentences, matching count from corpus used for vocabulary survey", total_examples)
            else:
                raise ValueError("you must provide either total_words or total_examples, to enable alpha and progress calculations")

        job_tally = 0

        if self.iter > 1:
            sentences = utils.RepeatCorpusNTimes(sentences, self.iter)
            total_words = total_words and total_words * self.iter
            total_examples = total_examples and total_examples * self.iter

        def worker_loop():
            """Train the model, lifting lists of sentences from the job_queue."""
            work = matutils.zeros_aligned(self.layer1_size, dtype=REAL)  # per-thread private work memory
            neu1 = matutils.zeros_aligned(self.layer1_size, dtype=REAL)
            jobs_processed = 0
            while True:
                job = job_queue.get()
                if job is None:
                    progress_queue.put(None)
                    break  # no more jobs => quit this worker
                sentences, alpha = job
                tally, raw_tally = self._do_train_job(sentences, alpha, (work, neu1))
                progress_queue.put((len(sentences), tally, raw_tally))  # report back progress
                jobs_processed += 1
#             logger.debug("worker exiting, processed %i jobs", jobs_processed)

        def job_producer():
            """Fill jobs queue using the input `sentences` iterator."""
            job_batch, batch_size = [], 0
            pushed_words, pushed_examples = 0, 0
            next_alpha = self.alpha
            job_no = 0

            for sent_idx, sentence in enumerate(sentences):
                sentence_length = self._raw_word_count([sentence])

                # can we fit this sentence into the existing job batch?
                if batch_size + sentence_length <= self.batch_words:
                    # yes => add it to the current job
                    job_batch.append(sentence)
                    batch_size += sentence_length
                else:
                    # no => submit the existing job
                    #logger.debug(
                    #    "queueing job #%i (%i words, %i sentences) at alpha %.05f",
                    #    job_no, batch_size, len(job_batch), next_alpha)
                    job_no += 1
                    job_queue.put((job_batch, next_alpha))

                    # update the learning rate for the next job
                    if self.min_alpha < next_alpha:
                        if total_examples:
                            # examples-based decay
                            pushed_examples += len(job_batch)
                            progress = 1.0 * pushed_examples / total_examples
                        else:
                            # words-based decay
                            pushed_words += self._raw_word_count(job_batch)
                            progress = 1.0 * pushed_words / total_words
                        next_alpha = self.alpha - (self.alpha - self.min_alpha) * progress
                        next_alpha = max(self.min_alpha, next_alpha)

                    # add the sentence that didn't fit as the first item of a new job
                    job_batch, batch_size = [sentence], sentence_length

            # add the last job too (may be significantly smaller than batch_words)
            if job_batch:
#                 logger.debug(
#                     "queueing job #%i (%i words, %i sentences) at alpha %.05f",
#                     job_no, batch_size, len(job_batch), next_alpha)
                job_no += 1
                job_queue.put((job_batch, next_alpha))

            if job_no == 0 and self.train_count == 0:
                logger.warning(
                    "train() called with an empty iterator (if not intended, "
                    "be sure to provide a corpus that offers restartable "
                    "iteration = an iterable)."
                )

            # give the workers heads up that they can finish -- no more work!
            for _ in xrange(self.workers):
                job_queue.put(None)
            logger.debug("job loop exiting, total %i jobs", job_no)

        # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
        job_queue = Queue(maxsize=queue_factor * self.workers)
        progress_queue = Queue(maxsize=(queue_factor + 1) * self.workers)

        workers = [threading.Thread(target=worker_loop) for _ in xrange(self.workers)]
        unfinished_worker_count = len(workers)
        workers.append(threading.Thread(target=job_producer))

        for thread in workers:
            thread.daemon = True  # make interrupting the process with ctrl+c easier
            thread.start()

        example_count, trained_word_count, raw_word_count = 0, 0, word_count
        start, next_report = default_timer() - 0.00001, 1.0
        prev_example_count = 0

        while unfinished_worker_count > 0:
            report = progress_queue.get()  # blocks if workers too slow
            if report is None:  # a thread reporting that it finished
                unfinished_worker_count -= 1
#                 logger.info("worker thread finished; awaiting finish of %i more threads", unfinished_worker_count)
                continue
            examples, trained_words, raw_words = report
            job_tally += 1

            # update progress stats
            example_count += examples
            trained_word_count += trained_words  # only words in vocab & sampled
            raw_word_count += raw_words

            # log progress once every report_delay seconds
            elapsed = default_timer() - start
            if elapsed >= next_report:
                next_report = elapsed + report_delay

        # all done; report the final stats
        elapsed = default_timer() - start
        logger.info(
            "training on %i raw words (%i effective words) took %.1fs, %.0f effective words/s",
            raw_word_count, trained_word_count, elapsed, trained_word_count / elapsed)
        if job_tally < 10 * self.workers:
            logger.warn("under 10 jobs per worker: consider setting a smaller `batch_words' for smoother alpha decay")

        # check that the input corpus hasn't changed during iteration
        if total_examples and total_examples != example_count:
            logger.warn("supplied example count (%i) did not equal expected count (%i)", example_count, total_examples)
        if total_words and total_words != raw_word_count:
            logger.warn("supplied raw word count (%i) did not equal expected count (%i)", raw_word_count, total_words)

        self.train_count += 1  # number of times train() has been called
        self.total_train_time += elapsed
        self.clear_sims()

        return trained_word_count
