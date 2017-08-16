#!/usr/bin/env python

from sklearn.metrics import mutual_info_score

def apply_mutinfo(client_trace, server_trace):
    scores = []
    for feature in xrange(0, len(client_trace[0])):
        try:
            feature_client = [row[feature] for row in client_trace]
            feature_server = [row[feature] for row in server_trace]
        except Exception as err:
            print 'Could not get feature from matrix for mutual info: ', err

        try:
            limitation = min(len(feature_client), len(feature_server))

            feature_client = feature_client[0:limitation]
            feature_server = feature_server[0:limitation]
        except Excetption as err:
            print 'Could not get min for Mutual Info: ', err

        try:
            score = mutual_info_score(feature_client, feature_server)
            scores.append(score)

        except Exception as err:
            print 'Error applying Pearson: ', err
            scores.append(0)

    return scores
