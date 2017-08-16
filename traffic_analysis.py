#!/usr/bin/env python
from twisted.enterprise import adbapi
from twisted.internet import defer

import logging

import numpy as np
from numpy import array

# metrics
from scalar_packet_count import apply_packet_count
from distance_pca_pearson import apply_pca_pearson
from distance_pearson import apply_pearson
from distance_rmse import apply_rmse
from distance_mutinfo import apply_mutinfo

import click

import time

class SetupParameters():
    """
    Organizes the relevant setup parameter we need later on to query the
    traces from the database.

    Constructor Arguments:
    num_reps: number of random repetitions available for an experiment. We will
        iterate through all repetitions to get more stable average results (and
        the standard deviation)
    num_clients: number of clients used in a setup. We need this number to
        iterate through the clients and servers to compare all the traces. We
        can derive the number of servers from the setup parameter below.
    setup: if we use a directed setup, each client connects to an individual
        server, so the number of servers is equal to the number of clients. In
        case of an undirected setup we only have two servers and the first group
        of clients connects to server 1, the second group connects to server 2.
        The number of clients in each group is identical.
    setup_index: we want to analyze all setups that are in the DB, so the we
        iterate over all indexes present in the setups table. setup_index is
        the current index
    features: is a string list of features that we want to use for the analysis,
        e.g., a subset of the 5 available traffic features from the parsing
    num_features: must be set compliant to the features string, should be replaced
        by a function returning the number of distinct features in the string.
    num_metrics: stores how many metrics we have in the framework right now.
        Must be changed as soon as a new metric is added, should be constant
        when we reach a status where a reasonable set of metrics is implemented.

    num_servers: depends on the setup that was used in the experiments. In the
        directed setup we have n:n connections from client to server, in the
        undirected/grouped setup we have n:2 connections to only 2 servers.
    """

    def __init__(self, num_reps, num_clients, setup, setup_index, features, num_features, num_metrics):
        self.setup = setup
        self.num_repetitions = num_reps
        self.num_clients = num_clients
        self.setup_index = setup_index
        self.features = features
        self.num_features = num_features
        self.num_metrics = num_metrics

        if setup == 'directed':
            self.num_servers = num_clients
        else:
            self.num_servers = 2

class ResultsContainer():
    """
    Keeps track of client results and repetition results.
    See examples below for specification.

    Client Results:
    - each client trace is compared with all server traces
    - all comparisons of this kind are stored in the client results
    - Example for 5 clients, 5 servers:
        - client_result = [c1 vs. s1, c1 vs. s2, ..., c1 vs. s5]
        - cx vs. cy means that the trace of client x is compared
          to the trace of server y
        - all comparison results are stored in the client result list
        - for each client a new client_result list is generated

    Repetition Results:
    - We have several random repetitions for each individual setup
    - Per repetition the results are stored in a list of Result objects
    - After each client was evaluated completely (compared to all servers)
      we make an identification guess for each metric and verify the correctness
    - According to this guess the num_corrects or num_fails counter in the
      Result object is incremented
    - After all clients are analyzed, the Result object is complete, and
      can be appended to the list of repetition results
    - After all repetitions are finished, we have complete results for
      a setup
    """
    def __init__(self, metrics_type):
        self.results_container = []
        self.type = metrics_type

    def append_item(self, item):
        """
        Argument item:
        - item can be either a float (metric result for client_results)
        - or a Result object for repetition_results
        """
        self.results_container.append(item)

    def get_length(self):
        cnt = 0
        for elem in self.results_container:
            cnt += 1
        return cnt

    def get_num_corrects(self):
        correct_counter = 0

        for result in self.results_container:
            correct_counter += result.num_corrects

        return correct_counter

    def get_num_fails(self):
        fail_counter = 0

        for result in self.results_container:
            fail_counter += result.num_fails

        return fail_counter

    def get_relative_success(self):
        relative_success_list = []

        for result in self.results_container:
            num_corrects = result.num_corrects
            num_fails = result.num_fails
            total = num_corrects + num_fails

            relative_success = float(num_corrects) / float(total)
            relative_success_list.append(relative_success)

        return relative_success_list

    def get_list_element(self, index):
        return self.results_container[index]

    def get_matrix_element(self, row, col):
        return self.results_container[row][col]

class Result():
    """
    Keeps track of the number of correct and wrong guesses in a repetition.
    The relative success can be computed from these.
    """
    def __init__(self):
        self.num_corrects = 0
        self.num_fails = 0

    def increment_corrects(self):
        self.num_corrects += 1

    def increment_fails(self):
        self.num_fails += 1

    def get_num_corrects(self):
        return self.num_corrects

    def get_num_fails(self):
        return self.num_fails

def verify_guesses(client_container, comparison_container, setup_parameters, client_index):
    """
    Verify the metrics results for
    """

    # node indexes start at 1, result arrays start at 0, decrement node indexes for comparison
    if setup_parameters.setup == 'directed':
        correct_target = client_index - 1
    else:
        if client_index < setup_parameters.num_servers:
            correct_target = 0
        else:
            correct_target = 1

    for metric in xrange(0,setup_parameters.num_metrics):
        for feature in xrange(0,setup_parameters.num_features):

            metric_feature_attempts = []
            for server in xrange(0, setup_parameters.num_servers):
                guess_matrix = comparison_container.get_list_element(server)

                try:
                    metric_feature_attempts.append(guess_matrix[metric][feature])
                except Exception as err:
                    print 'Problem appending attempts ', err

                try:
                    tmp_guess = client_container.get_matrix_element(metric, feature)
                except Exception as err:
                    print 'Problem getting tmp_guess thingy', err

            if metric == 0 or metric == 3:
                # scalar (0)
                # rmse (3)
                try:
                    min_index = min(enumerate(metric_feature_attempts),key=lambda x: x[1])[0]
                except Exception as err:
                    print 'Problem getting min index for scalar or rmse ', err

                if min_index == correct_target:
                    tmp_guess.increment_corrects()
                else:
                    tmp_guess.increment_fails()

            elif metric == 1 or metric == 2 or metric == 4:
                # pca + pearson (1)
                # pearson (2)
                # mutual info (4)
                try:
                    max_index = max(enumerate(metric_feature_attempts),key=lambda x: x[1])[0]
                except Exception as err:
                    print 'Problem getting max index for pcap, pearson, or mutual info ', err

                if max_index == correct_target:
                    tmp_guess.increment_corrects()
                else:
                    tmp_guess.increment_fails()
            else:
                # not covered right now
                print 'Not covered, what happened?'

@defer.inlineCallbacks
def analyze_repetitions(reactor, dbpool, setup_parameters, db_name):
    """
    List of results for each repetition. Entries are matrices of the following format:

    Rows:
        Metrics
    Columns:
        Features (see setup_parameters.features)
    """

    print 'Analyze Repetitions'
    repetitions_container = ResultsContainer('repetition results')

    for repetition in xrange(1, setup_parameters.num_repetitions + 1):
        print 'Repetition {}/{}'.format(repetition, setup_parameters.num_repetitions)
        """
        Matrix of Result objects to save the results of the identification attack.
        Entries are Result objects and get incremented after each client iteration,
        organized as follows:

        Rows:
            Metrics
        Columns:
            Features
        """
        try:
            repetition_ok = yield dbpool.runQuery('SELECT EXISTS(SELECT * FROM {}.traces_submission WHERE setup_id={} AND repetition={});'.format(
                db_name,
                setup_parameters.setup_index,
                repetition
            ))
        except Exception as err:
            print err

        if repetition_ok[0][0] == 1:

            client_container = ResultsContainer('client results')
            for metric in xrange(0,setup_parameters.num_metrics):
                tmp_list = []
                for feature in xrange(0,setup_parameters.num_features):
                    result = Result()
                    tmp_list.append(result)
                client_container.append_item(tmp_list)

            for client_index in xrange(1,setup_parameters.num_clients + 1):
                print 'Client {}/{}'.format(client_index, setup_parameters.num_clients)
                """
                Matrix of floats from the comparison of one client with all servers.
                Entries are the results of applied metrics. Rows and Columns like before.
                """
                try:
                    client_trace = yield dbpool.runQuery('SELECT {features} FROM {database}.traces_submission WHERE repetition={rep} AND setup_id={setup_id} AND node_id={node_index};'.format(
                            features = setup_parameters.features,
                            database = db_name.
                            rep = repetition,
                            setup_id = setup_parameters.setup_index,
                            node_index = client_index
                        ))
                except Exception as err:
                    print 'Problem querying client trace: ', err

                if len(client_trace) == 0:
                    print 'Skipped Client ', client_index
                    break

                comparison_container = ResultsContainer('server comparisons')

                server_id_string = '('
                for server_index in  xrange(31,setup_parameters.num_servers + 31):
                    server_id_string += 'node_id = ' + str(server_index)
                    if server_index < setup_parameters.num_servers + 30:
                        server_id_string += ' OR '

                server_id_string += ')'

                server_features = 'node_id, ' + setup_parameters.features

                try:
                    all_server_traces = yield dbpool.runQuery('SELECT {features} FROM {database}.traces_submission WHERE repetition={rep} AND setup_id={sid} AND {server_ids};'.format(
                        features = server_features,
                        database = db_name,
                        rep = repetition,
                        sid = setup_parameters.setup_index,
                        server_ids = server_id_string
                    ))
                except Excetption as err:
                    print 'Problem querying all servers: ', err

                server_fail = 0
                for server_index in  xrange(31,setup_parameters.num_servers + 31):

                    server_trace = filter(lambda x: x[0]==server_index, all_server_traces)

                    if len(server_trace) == 0:
                        print 'Skipped Server ', server_index
                        server_fail = 1
                        break

                    res_scalar_pc = apply_packet_count(client_trace, server_trace)
                    res_dist_pcap = apply_pca_pearson(client_trace, server_trace)
                    res_dist_p = apply_pearson(client_trace, server_trace)
                    res_dist_rmse = apply_rmse(client_trace, server_trace)
                    res_dist_mutinfo = apply_mutinfo(client_trace, server_trace)

                    metric_feature_matrix = [res_scalar_pc, res_dist_pcap, res_dist_p, res_dist_rmse, res_dist_mutinfo]
                    comparison_container.append_item(metric_feature_matrix)

                if server_fail == 1:
                    print 'Skipped client because server was corrupt'
                    break

                verify_guesses(client_container, comparison_container, setup_parameters, client_index)

            repetitions_container.append_item(client_container)

    yield write_ta_results(reactor, dbpool, repetitions_container, setup_parameters, db_name)

@defer.inlineCallbacks
def write_ta_results(reactor, dbpool, results_container, setup_params, db_name):
    """
    Get the average and standard deviation for repetitions and write these
    aggregated results to the database.
    """

    print 'Write Results'

    db_columns = '(setup_id, metric, feature, success_avg, success_sd, num_successes, num_fails)'
    id_string = '"{}"'.format(setup_params.setup_index)

    metric_strings = ['"scalar_counts"', '"distance_pca_pearson"', '"distance_pearson"', '"distance_rmse"', '"distance_mutinfo"']
    feature_strings = ['"packet_counts"', '"inter_arrival_time"', '"packet_length"', '"time_to_live"', '"window_size"']

    for metric in xrange(0, setup_params.num_metrics):
        metric_string = metric_strings[metric]

        for feature in xrange(0, setup_params.num_features):
            feature_string = feature_strings[feature]

            corrects = []
            total_corrects = []
            fails = []
            total_fails = []

            for repetition in xrange(0, results_container.get_length()):

                client_results = results_container.get_list_element(repetition)
                results_object = client_results.get_matrix_element(metric, feature)

                num_corrects = results_object.get_num_corrects()
                num_fails = results_object.get_num_fails()

                total_guesses = num_corrects + num_fails

                if total_guesses > 0:
                    rel_correct = float(num_corrects) / float(total_guesses)
                    rel_fail = float(num_fails) / float(total_guesses)
                else:
                    rel_correct = 0
                    rel_fail = 0

                total_corrects.append(num_corrects)
                total_fails.append(num_fails)

                corrects.append(rel_correct)
                fails.append(rel_fail)

            try:
                avg_correct = sum(corrects) / len(corrects)
                sd_correct = np.std(corrects)
            except Exception as err:
                print 'Problem getting avg corrects: ', err
                avg_correct = 0
                sd_correct = 0

            total_correct = sum(total_corrects)
            total_fail = sum(total_fails)

            avg_correct_string = '"{}"'.format(avg_correct)
            sd_correct_string = '"{}"'.format(sd_correct)
            total_correct_string = '"{}"'.format(total_correct)
            total_fail_string = '"{}"'.format(total_fail)

            try:
                yield dbpool.runQuery('INSERT INTO {}.ta_submission {} VALUES ({},{},{},{},{},{},{});'.format(
                    db_name,
                    db_columns,
                    setup_params.setup_index,
                    metric_string,
                    feature_string,
                    avg_correct_string,
                    sd_correct_string,
                    total_correct_string,
                    total_fail_string
                ))
            except Exception as err:
                print 'Problem writing Results to DB: ', err

@defer.inlineCallbacks
def analyze_setups(reactor, dbpool, select_setup, db_name):
    """
    Iterate through all setups in the db_name.setups table and apply the
    set of metrics to it.
    """
    if select_setup == 'All' or select_setup == 'all':
        try:
            start_time = int(round(time.time() * 1000))
            num_setups = yield dbpool.runQuery('SELECT COUNT(*) FROM {}.setups_submission;'.format(db_name))
            end_time = int(round(time.time() * 1000))
            print 'Query Setup ', end_time - end_time

        except Exception as err:
            print 'Problem in select count: ', err

        num_setups = num_setups[0][0]
    else:
        num_setups = 1
        current_setup = select_setup

    for setup_index in xrange(1, num_setups + 1):
        if select_setup == 'All' or select_setup == 'all':
            print 'Setup {}/{}'.format(setup_index, num_setups)
            current_setup = setup_index
        else:
            print 'Setup ', current_setup

        try:
            setup_data = yield dbpool.runQuery('SELECT setup, num_clients, repetitions FROM {}.setups_submission WHERE id = {};'.format(db_name, current_setup))
        except Exception as err:
            print 'Error querying setup_data: ', err

        network_setup = setup_data[0][0]
        num_clients = setup_data[0][1]
        num_reps = setup_data[0][2]

        features = 'packet_count, inter_arrival_time, packet_length, time_to_live, window_size'

        num_features = 5
        num_metrics = 5

        setup_params = SetupParameters(num_reps, num_clients, network_setup, current_setup, features, num_features, num_metrics)

        yield analyze_repetitions(reactor, dbpool, setup_params, db_name)

@click.command()
@click.option('--select-setup', default=None, type=str, help='Enter Setup ID if you want a specific setup, "All" otherwise')
@click.option('--db-name', default=None, type=str, help='Name of DB')
@click.option('--db-user', default=None, type=str, help='Username DB')
@click.option('--db-passwd', default=None, type=str, help='Password DB')
@click.option('--db-port', default=None, type=int, help='DB connection port')
@click.option('--db-host', default=None, type=str, help='DB connection host')
def main(select_setup, db_name, db_user, db_passwd, db_port, db_host):

    try:
        dbpool = adbapi.ConnectionPool('MySQLdb', host=db_host, db=db_name, user=db_user, passwd=db_passwd, port=db_port)
    except Exception as err:
        print 'Failed to connect to DB:', err

    from twisted.internet import reactor

    deferred = analyze_setups(reactor, dbpool, select_setup, db_name)
    deferred.addCallback(lambda ign: reactor.stop())

    reactor.run()

if __name__ == '__main__':
    main()
