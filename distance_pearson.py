#!/usr/bin/env python

from numpy.ma import corrcoef

def apply_pearson(client_trace, server_trace):

    coeffs = []
    for feature in xrange(0, len(client_trace[0])):
        feature_client = [row[feature] for row in client_trace]
        feature_server = [row[feature] for row in server_trace]

        limitation = min(len(feature_client), len(feature_server))

        feature_client = feature_client[0:limitation]
        feature_server = feature_server[0:limitation]
        
        try:
            cor = corrcoef(feature_client, feature_server)
            correlation_coefficient = abs(cor.data[1][0])
            coeffs.append(correlation_coefficient)

        except Exception as err:
            print 'Error applying Pearson: ', err
            coeffs.append(0)

    return coeffs