#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# 
# Date: 24.12.2011
#
#
from __future__ import division

import argparse
import cPickle as pickle
from cStringIO import StringIO
import string

import scipy as sp
import numpy as np
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Scaler
from sklearn.utils import shuffle

from app_libs.db_proc import start_db, stop_db, AnT
from app_libs.learners import *


samples_object = None
learners = None


def load_samples(file_name):
    global samples_object
    samples_object = pickle.load(open(file_name, "rb"))
    labels = []
    gains = []
    losses = []
    trans_marks = []
    for el in samples_object["sample_ids"]:
        an = AnT.get(el)
        labels.append(an.class_label)
        gains.append(an.success_gain)
        losses.append(an.error_cost)
        if el in samples_object["trans_sample_ids"]:
            trans_marks.append(True)
        else:
            trans_marks.append(False)
    classes = [1 if x == "pos" else -1 for x in labels]
    return (samples_object["matrix"], np.array(classes),
        np.array(gains), np.array(losses),
        np.array(trans_marks, dtype=np.bool))


def output_param(pname, p, k, perc=True):
    pmean = p.mean(0)
    pstd = p.std(0) / 2
    if perc:
        pmean *= 100
        pstd *= 100
        st = "   {:<20}: pos: {:>6.2f} (+/- {:=5.2f})% neg: \
{:>6.2f} (+/- {:=5.2f})%".format(pname, pmean[k, 0], pstd[k, 0],
pmean[k, 2], pstd[k, 2])
    else:
        st = "   {:<20}: pos: {:>6.2f} (+/- {:=5.2f})  neg: \
{:>6.2f} (+/- {:=5.2f})".format(pname, pmean[k, 0], pstd[k, 0],
pmean[k, 2], pstd[k, 2])
    print st


def make_check(samples, classes, gain, loss, W, markers, sparse):
    n_samples = samples.shape[0]
    n_features = samples.shape[1]

    only_train_indxs = np.arange(n_samples)[markers]
    S = samples
    C = classes
    G = gain
    L = loss
    if W is not None:
        weights = True
    else:
        weights = False

    cv = cross_validation.StratifiedKFold(C, 10, indices=True)

    precision = np.zeros((cv.n_folds, len(learner_names), 3))
    precision_total = np.zeros((cv.n_folds, len(learner_names)))
    recall = np.zeros((cv.n_folds, len(learner_names), 3))
    support = np.zeros((cv.n_folds, len(learner_names), 3))
    gain = np.zeros((cv.n_folds, len(learner_names), 3))
    nd_freq = np.zeros((cv.n_folds, len(learner_names), 3))
    total = np.zeros((cv.n_folds, len(learner_names), 3))
    conf_m = np.zeros((2, 2, len(learner_names)), dtype=int)

    res = [0] * len(learner_names)

    for i, (train, test) in enumerate(cv):
        test = np.array([x for x in test if x not in only_train_indxs])
        print "Iteration: %d of %d" % (i+1, cv.n_folds)
        real = C[test]
        prediction = []

        if weights:
            trS, trC, trW = shuffle(S[train], C[train], W[train])
        else:
            trS, trC = shuffle(S[train], C[train])

        if sparse:
            scaler = None
            # scaler = Scaler(with_mean=False).fit(trS)
            trS = trS
            tsS = S[test]
        else:
            scaler = Scaler(with_mean=True).fit(trS)
            trS = scaler.transform(trS)
            tsS = scaler.transform(S[test])

        for k, learner in enumerate(learners):
            print "\tLearner: %s" % learner_names[k]
            if k == 5 and sparse:
                # Last learner (boosting) only supports dense data
                learner.fit(trS.todense(), trC)
                pr = learner.predict(tsS.todense())
            else:
                # k-Nearest & GradientBoosting has no sample_weight
                if weights and k != 3 and k != 5: 
                    learner.fit(trS, trC, sample_weight=trW)
                else:
                    learner.fit(trS, trC)
                pr = learner.predict(tsS)
            res[k] = pr
        #
        # Always pos classifier
        res[k+4]= np.ones(len(test))
        # Always neg classifier
        res[k+5] = -1 * np.ones(len(test))
        # Random classifier
        res[k+6] = np.random.randint(0, 2, len(test)) * 2 - 1

        # Agregated classifiers
        agr = np.sum(res[k] for k in range(len(learners)))
        d = 2 * (agr > 0) - 1
        # Here could be -1, 0, 1; 0 means no deceision
        res[k+1] = d * (np.abs(agr) > 0) # majority
        res[k+2] = d * (np.abs(agr) > 3)
        res[k+3] = d * (np.abs(agr) == len(learners)) # all

        for k, name in enumerate(learner_names):
            pr = res[k]

            (p, r, f1, s) = metrics.precision_recall_fscore_support(real,
                pr, labels=[1, 0, -1])

            conf_m[:, :, k] += metrics.confusion_matrix(real, pr, labels=[1, -1])

            precision[i, k] = p
            recall[i, k] = r
            support[i, k] = s
            precision_total[i, k] = (p[0]*s[0] + p[2]*s[2])/(s[0]+s[2])

            nd_freq[i, k, 0] = np.sum((pr == 0) * (real == 1)) / s[0]
            nd_freq[i, k, 1] = 0.0
            nd_freq[i, k, 2] = np.sum((pr == 0) * (real == -1)) / s[2]

            total[i, k, 0] = sum((pr == 1))
            total[i, k, 1] = sum((pr == 0))
            total[i, k, 2] = sum((pr == -1))

            ev_pos = pr * real * (pr == 1) # 1 - ok, -1 - error
            ev_neg = pr * real * (pr == -1) # 1 - ok, -1 - error

            gain_pos = G[test].dot(ev_pos > 0)
            gain_neg = G[test].dot(ev_neg > 0)
            loss_pos = L[test].dot(ev_pos < 0)
            loss_neg = L[test].dot(ev_neg < 0)

            gain[i, k, 0] = gain_pos - loss_pos
            gain[i, k, 1] = 0.0
            gain[i, k, 2] = gain_neg - loss_neg

    print "-------------------------\n"
    print "RESULTS"
    for k, name in enumerate(learner_names):
        print "Learner: %s" % name
        print "   Total precision: %f (+/- %f)%%" %\
            (100*precision_total.mean(0)[k], 100*precision_total.std(0)[k]/2)
        output_param("Precision   ", precision, k)
        output_param("Recall      ", recall, k)
        output_param("Support     ", support, k, perc=False)
        output_param("Supposed    ", total, k, perc=False)
        output_param("No decision ", nd_freq, k)
        output_param("Gain total", gain, k, perc=False)
        output_param("Gain per supposition", gain/total, k, perc=False)
        print "   Confusion matrix:"
        my_matrix = np.matrix(conf_m[:, :, k]).tolist()
        my_matrix[0] = ['pos'] + my_matrix[0]
        my_matrix[1] = ['neg'] + my_matrix[1]
        my_matrix = [[' ', 'pos', 'neg']] + my_matrix
        max_lens = [max([len(str(r[i])) for r in my_matrix])
            for i in range(len(my_matrix[0]))]

        print "\n".join(["".join([string.ljust(str(e), l + 2)
            for e, l in zip(r, max_lens)]) for r in my_matrix])
        #print conf_m[:, :, k]

    return (precision, recall, support, gain, nd_freq, total, precision_total)


def main():
    global learners
    parser = argparse.ArgumentParser(
        description="Learn and test")
    parser.add_argument("samples_file",
        metavar="SamplesFile",
        type=str,
        nargs=1,
        help="File with samples. See documentation for details")
    parser.add_argument("--save-model", action="store",
        help="File name for model save. If file already exists\
            it will be rewritten. If not defined then model will\
            not be saved")
    parser.add_argument("--use-loss", action="store_true",
        help="Use loss as sample weight for training")
    parser.add_argument("--use-gain", action="store_true",
        help="Use gain as sample weight for training (ignored\
            if --use-loss used)")
    parser.add_argument("--no-test", action="store_true",
        help="Don't make crossvalidation, just train and save model")

    args = parser.parse_args()
    start_db()

    (S, C, G, L, M) = load_samples(args.samples_file[0])
    if S.__class__ == sp.sparse.csr.csr_matrix:
        learners = learners_sparse
        sparse = True
    elif S.__class__ == np.ndarray:
        learners = learners_dense
        sparse = False
    else:
        print "Wrong matrix format"
        return

    #to_load = pickle.load(open("l.p", "rb"))
    #learners = to_load["learners"]

    if args.use_loss:
        W = L
    elif args.use_gain:
        W = G
    else:
        W = None
    if not args.no_test:
        make_check(S, C, G, L, W, M, sparse)
    if args.save_model:
        print
        print "Training on all samples\n------------------------"
        if sparse:
            scaler = None
            #scaler = Scaler(with_mean=False).fit(S)
            S = S
        else:
            scaler = Scaler(with_mean=True).fit(S)
            S = scaler.transform(S)

        for k, learner in enumerate(learners):
            print "Training %s" % learner_names[k]
            if k == 5 and sparse:
                learner.fit(S.todense(), C)
                # Last learner (boosting) only supports dense data
            else:
                if W is None or k == 3 or k == 5:
                    learner.fit(S, C)
                else:
                    learner.fit(S, C, sample_weight=W)
        print "Saving..."
        to_save = {}
        to_save["trans_file"] = samples_object["trans_file"]
        to_save["learners"] = learners
        to_save["scaler"] = scaler
        pickle.dump(to_save, open(args.save_model, "wb"))
        print "Done"
    return

if __name__ == "__main__":
    r = main()
