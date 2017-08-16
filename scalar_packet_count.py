#!/usr/bin/env python

def apply_packet_count(client_trace, server_trace):
    """
    Simple scalar comparison where the sum of each column is compared
    for pairs of client and server.
    """
    diffs = []

    for feature in xrange(0,len(client_trace[0])):
        summed_client = sum([row[feature] for row in client_trace])
        summed_server = sum([row[feature] for row in server_trace])

        try:
            diff = abs(summed_client - summed_server)
        except Exception as err:
            print err
        diffs.append(diff)

    return diffs
