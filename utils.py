import numpy as np


def action_validity(action_idxs, availability_matrix):
    return availability_matrix[action_idxs[0], action_idxs[1], action_idxs[2]]


def calculate_earliest_times(operation_index, aircraft_info):
    eta = aircraft_info['ETA']



