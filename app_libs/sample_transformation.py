#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Author: Alexice
# Date: 05.12.2011
#
# Parse options of transformation
from __future__ import division

import cPickle as pickle
from cStringIO import StringIO
from itertools import chain

from nltk.corpus import stopwords
from nltk import FreqDist, DictionaryProbDist
import numpy as np
from scipy.sparse import *
from scipy import *
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.text import TfidfTransformer


def tr_load(file_name):
    return pickle.load(open(file_name, "rb"))


def tr_save(file_name, trans):
    pickle.dump(trans, open(file_name, "wb"))


class TransError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Transformation(object):
    """
    Base and yet the only class for text to vector transformations
    """

    def __init__(self, sq, trans_opts=None):
        # Announcement ids which were taken for transformation creation
        self.ids = []
        # Number of words in dictionary
        self.N = 0
        # Number of announcements processed = len(ids)
        self.n_docs = 0
        # List of words with counts and idfs
        self.dwords = []
        # Dictionary for fast word search
        self.dct = {}
        # Number of output dimensions if reduction is used
        self.dim = 0
        if trans_opts:
            self.opts = trans_opts
            self._generate(sq)
        else:
            raise TransError("Wrong init parameters")

    def get_term_document_matrix(self, sq):
        matrix = self._get_term_document_matrix(sq)
        if self.opts["reduction"]:
            matrix = self.decomposition.transform(matrix)
        return matrix

    def get_sample_vector(self, txt):
        vector = self._get_sample_vector(txt)
        if self.opts["reduction"]:
            vector = self.decomposition.transform(vector)
        return vector

    def _extract_negations(self, ttokens, extract=False):
        if extract:
            tokens = []
            i, l = 0, len(ttokens)
            while i<l:
                word = ttokens[i].lower()
                if word == "not" and l > i+1:
                    tokens.append("not " + ttokens[i+1].lower())
                    i += 2
                else:
                    tokens.append(word)
                    i += 1
        else:
            tokens = ttokens
        return tokens

    def _get_bag(self, txt, tfidf=False):
        """
        txt should be list of lists!
        Returns just counts
        """
        # Negations
        tokens = self._extract_negations([item for sublist in
            txt for item in sublist], extract=self.opts["neg"])
        fd = FreqDist(tokens)
        return [(x, fd[x]) for x in fd.samples() if x in self.dct]

    def _get_term_document_matrix(self, sq, sett=False):
        matrix = self._get_term_count_matrix(sq)
        if self.opts["boolean"]:
            (ii, jj) = matrix.nonzero()
            for i, j in zip(ii, jj):
                matrix[i, j] = 1.0
            return matrix
        if sett:
            # Call for t calculation
            self.t = TfidfTransformer(norm=None, use_idf=False,
                smooth_idf=False)
            if self.opts["tfidf"]:
                self.t.set_params(use_idf=True, smooth_idf=True)
            if self.opts["normalize"]:
                self.t.set_params(norm="l2")
            self.t.fit(matrix)
        if self.t:
            return self.t.transform(matrix)
        else:
            # Returns just count matrix
            return matrix

    def _get_term_count_matrix(self, sq):
        M = sq.count()
        N = self.N
        data = []
        row = []
        col = []
        for j, an in enumerate(sq):
            bag = self._get_bag(an.text_tokenized)
            for word, value in bag:
                i = self.dct[word][0]
                data.append(value)
                col.append(i)
                row.append(j)
        return csr_matrix((data, (row, col)), shape=(M, N), dtype=np.float64)

    def _get_sample_vector(self, txt):
        """
        Returns sample vector of frequencies, occurences or tfidf
        transformed and normalized if optioned
        """
        bag = self._get_bag(txt)
        vector = np.zeros((1, self.N), dtype=np.float)
        for word, value in bag:
            pos = self.dct[word][0]
            vector[0, pos] = value
        if self.opts["boolean"]:
            return 1.0*(vector != 0)
        return self.t.transform(vector) #.todense()

    def _generate(self, sq):
        """
        Generate transformation based on transformation options
        """
        distr_pos = FreqDist()
        distr_neg = FreqDist()
        cnt_pos = 0
        cnt_neg = 0
        labels = []
        cnt = int(sq.count() * self.opts["ratio"])
        print "Processin ~ %d +/- %d records" % (cnt, sqrt(cnt))
        cnt = 0
        for j, an in enumerate(sq):
            if rand() > self.opts["ratio"]:
                continue
            cnt += 1
            self.ids.append(an.id)
            # Negations
            tokens = self._extract_negations([item for sublist in
                an.text_tokenized for item in sublist],
                extract=self.opts["neg"])

            # Stopwords and minimum token length
            if self.opts["stop"]:
                stop = stopwords.words("english")
                tokens = [w for w in tokens if w not in stop and
                    len(w)>=self.opts["min_tlen"]]
            else:
                tokens = [w for w in tokens if len(w)>=self.opts["min_tlen"]]

            # Calculating number of documents each token occures
            # For each class
            if an.class_label == "pos":
                cnt_pos += 1
                for w in FreqDist(tokens).samples():
                    distr_pos.inc(w)
            else:
                cnt_neg += 1
                for w in FreqDist(tokens).samples():
                    distr_neg.inc(w)

        self.n_docs = cnt
        for smp in chain(distr_pos, distr_neg):
            if distr_pos[smp] + distr_neg[smp] < self.opts["min_tokens"]:
                continue
            else:
                # +1 for idf smoothing
                idf_pos = np.log((cnt_pos+1)/(distr_pos[smp]+1))
                idf_neg = np.log((cnt_neg+1)/(distr_neg[smp]+1))
                rec = (smp, idf_pos, idf_neg, distr_pos[smp], distr_neg[smp])
                # Ensure no dupls
                if rec not in self.dwords:
                    # Add to list
                    self.dwords.append(rec)
        # Sort by pos/neg idf difference
        self.dwords.sort(key=lambda x: x[1]-x[2], reverse=True)
        if (self.opts["dict_cut"] > 0 and
            len(self.dwords) > 2*self.opts["dict_cut"]+1):
            self.dwords = self.dwords[:self.opts["dict_cut"]] +\
                self.dwords[-self.opts["dict_cut"]:]

        # Make dictionary for fast word search
        self.dct = dict([(x[0], (i, x[1], x[2]))
            for i, x in enumerate(self.dwords)])
        self.N = len(self.dwords)
        print "Words in dictionary: %d" % (self.N)
        print "Documents pos/neg/total: %d/%d/%d" %\
            (cnt_pos, cnt_neg, cnt_pos + cnt_neg)
        if not self.opts["boolean"] or self.opts["reduction"]:
            # This call is necessary for t creation, we don't use t
            # for boolean matrixes
            # But we need this matrix anyway if we want to make reduction
            matrix = self._get_term_document_matrix(sq, sett=True)
        if self.opts["reduction"]:
            print "Making reduction..."
            self.decomposition = RandomizedPCA(
                n_components=self.opts["components"], whiten=True)
            self.decomposition.fit(matrix)
            self.dim = min(matrix.shape, self.opts["components"])
            print "Output vectors are SVD components, dimension: %d" %\
                self.dim
        elif self.opts["boolean"]:
            print "Output vectors are words occurrences, dimension: %d" %\
                self.N
        else:
            print "Output vectors are words frequencies, dimension: %d" %\
                self.N

        print "Done"
