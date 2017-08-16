#!/usr/bin/env python

from sklearn.metrics import mean_squared_error
from math import sqrt

def apply_rmse(client_trace, server_trace):

    errors = []
    for feature in xrange(0, len(client_trace[0])):
        feature_client = [row[feature] for row in client_trace]
        feature_server = [row[feature] for row in server_trace]

        limitation = min(len(feature_client), len(feature_server))

        feature_client = feature_client[0:limitation]
        feature_server = feature_server[0:limitation]
        
        try:
            error = sqrt(mean_squared_error(feature_client, feature_server))
            errors.append(error)

        except Exception as err:
            print 'Error applying Pearson: ', err
            errors.append(0)

    return errors