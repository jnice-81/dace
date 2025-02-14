# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Implementation of the performance instrumentation report. """

import json
import numpy as np
import re


class InstrumentationReport(object):
    @staticmethod
    def get_event_uuid(event):
        uuid = (-1, -1, -1)
        if 'args' in event:
            args = event['args']
            if 'sdfg_id' in args and args['sdfg_id'] is not None:
                uuid = (args['sdfg_id'], -1, -1)
                if 'state_id' in args and args['state_id'] is not None:
                    uuid = (uuid[0], args['state_id'], -1)
                    if 'id' in args and args['id'] is not None:
                        uuid = (uuid[0], uuid[1], args['id'])
        return uuid

    def __init__(self, filename: str):
        # Parse file
        match = re.match(r'.*report-(\d+)\.json', filename)
        self.name = match.groups()[0] if match is not None else 'N/A'

        self.durations = {}
        self.counters = {}
        self._sortcat = None
        self._sortdesc = False

        with open(filename, 'r') as fp:
            report = json.load(fp)

            if 'traceEvents' not in report or 'sdfgHash' not in report:
                print(filename, 'is not a valid SDFG instrumentation report!')
                return

            self.sdfg_hash = report['sdfgHash']

            events = report['traceEvents']

            for event in events:
                if 'ph' in event:
                    phase = event['ph']
                    name = event['name']
                    if phase == 'X':
                        uuid = self.get_event_uuid(event)
                        if uuid not in self.durations:
                            self.durations[uuid] = {}
                        if name not in self.durations[uuid]:
                            self.durations[uuid][name] = []
                        self.durations[uuid][name].append(event['dur'] / 1000)
                    if phase == 'C':
                        if name not in self.counters:
                            self.counters[name] = 0
                        self.counters[name] += event['args'][name]

    def __repr__(self):
        return 'InstrumentationReport(name=%s)' % self.name

    def sortby(self, column: str, ascending: bool = False):
        if (column and column.lower() not in ('counter', 'value', 'min', 'max', 'mean', 'median')):
            raise ValueError('Only Counter, Value, Min, Max, Mean, Median are ' 'supported')
        self._sortcat = column if column is None else column.lower()
        self._sortdesc = not ascending

    def _get_runtimes_string(self,
                             label,
                             runtimes,
                             element,
                             sdfg,
                             state,
                             string,
                             row_format,
                             colw,
                             with_element_heading=True):
        indent = ''
        if len(runtimes) > 0:
            element_label = ''
            if element[0] > -1 and element[1] > -1 and element[2] > -1:
                # This element is a node.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
                sdfg = element[0]
                if state != element[1]:
                    # No parent state row present yet, print it.
                    string += row_format.format('|-State (' + str(element[1]) + ')', '', '', '', '', width=colw)
                state = element[1]
                element_label = '| |-Node (' + str(element[2]) + ')'
                indent = '| | |'
            elif element[0] > -1 and element[1] > -1:
                # This element is a state.
                if sdfg != element[0]:
                    # No parent SDFG row present yet, print it.
                    string += row_format.format('SDFG (' + str(element[0]) + ')', '', '', '', '', width=colw)
                sdfg = element[0]
                state = element[1]
                element_label = '|-State (' + str(element[1]) + ')'
                indent = '| |'
            elif element[0] > -1:
                # This element is an SDFG.
                sdfg = element[0]
                state = -1
                element_label = 'SDFG (' + str(element[0]) + ')'
                indent = '|'
            else:
                element_label = 'N/A'

            if with_element_heading:
                string += row_format.format(element_label, '', '', '', '', width=colw)

            string += row_format.format(indent + label + ':', '', '', '', '', width=colw)
            string += row_format.format(indent,
                                        '%.3f' % np.min(runtimes),
                                        '%.3f' % np.mean(runtimes),
                                        '%.3f' % np.median(runtimes),
                                        '%.3f' % np.max(runtimes),
                                        width=colw)

        return string, sdfg, state

    def getkey(self, element):
        events = self.durations[element]
        result = []
        for event in events.keys():
            runtimes = events[event]
            result.extend(runtimes)

        result = np.array(result)
        if self._sortcat == 'min':
            return np.min(result)
        elif self._sortcat == 'max':
            return np.max(result)
        elif self._sortcat == 'mean':
            return np.mean(result)
        else:  # if self._sortcat == 'median':
            return np.median(result)

    def __str__(self):
        COLW = 15
        COUNTER_COLW = 39

        element_list = list(self.durations.keys())
        element_list.sort()

        row_format = ('{:<{width}}' * 5) + '\n'
        counter_format = ('{:<{width}}' * 2) + '\n'

        string = 'Instrumentation report\n'
        string += 'SDFG Hash: ' + self.sdfg_hash + '\n'

        if len(self.durations) > 0:
            string += ('-' * (COLW * 5)) + '\n'
            string += ('{:<{width}}' * 2).format('Element', 'Runtime (ms)', width=COLW) + '\n'
            string += row_format.format('', 'Min', 'Mean', 'Median', 'Max', width=COLW)
            string += ('-' * (COLW * 5)) + '\n'

            sdfg = -1
            state = -1

            if self._sortcat in ('min', 'mean', 'median', 'max'):
                element_list = sorted(element_list, key=self.getkey, reverse=self._sortdesc)

            for element in element_list:
                events = self.durations[element]
                if len(events) > 0:
                    with_element_heading = True
                    for event in events.keys():
                        runtimes = events[event]

                        string, sdfg, state = self._get_runtimes_string(event, runtimes, element, sdfg, state, string,
                                                                        row_format, COLW, with_element_heading)
                        with_element_heading = False

            string += ('-' * (COLW * 5)) + '\n'

        if len(self.counters) > 0:
            string += ('-' * (COUNTER_COLW * 2)) + '\n'
            string += ('{:<{width}}' * 2).format('Counter', 'Value', width=COUNTER_COLW) + '\n'
            string += ('-' * (COUNTER_COLW * 2)) + '\n'

            if self._sortcat == 'value':
                counter_list = sorted(self.counters, key=lambda k: self.counters[k], reverse=self._sortdesc)
            elif self._sortcat == 'counter':
                counter_list = sorted(self.counters.keys(), reverse=self._sortdesc)
            else:
                counter_list = self.counters.keys()

            for counter in counter_list:
                string += counter_format.format(counter, self.counters[counter], width=COUNTER_COLW)
            string += ('-' * (COUNTER_COLW * 2)) + '\n'

        return string
