#!/usr/bin/env python

from sklearn import decomposition
from numpy.ma import corrcoef
from numpy import transpose

def apply_pca_pearson(client_trace, server_trace):
    """
    Applies PCA to input data and compares transformed data via
    Pearson correlation.
    """

    coeffs = []
    n_components=3
    
    for feature in xrange(0, len(client_trace[0])):
        tmp_coeff = 0
        try:
            pca = decomposition.PCA(n_components)
            pca.fit(client_trace)
            client_pca = pca.transform(client_trace)

            pca.fit(server_trace)
            server_pca = pca.transform(server_trace)
        
        except Exception as err:
            print 'Problem applying PCA: ', err
            
        try:
            for i in xrange(0,n_components):
                shrinked_client = client_pca[0:1000]
                shrinked_server = server_pca[0:1000]

                shrinked_client = [row[i] for row in shrinked_client]
                shrinked_server = [row[i] for row in shrinked_server]

                limitation = min(len(shrinked_client), len(shrinked_server))

                shrinked_client = shrinked_client[0:limitation]
                shrinked_server = shrinked_server[0:limitation]

                cor = corrcoef(transpose(shrinked_client), transpose(shrinked_server))
                correlation_coefficient = abs(cor.data[1][0])

                if correlation_coefficient > tmp_coeff:
                    tmp_coeff = correlation_coefficient

        except Exception as err:
            print 'Error applying PCA-Pearson: ', err
	    
        coeffs.append(tmp_coeff)

    return coeffs
