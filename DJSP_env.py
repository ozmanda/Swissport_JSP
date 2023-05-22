import numpy as np
import pandas as pd
from pandas import to_datetime, Timedelta
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete
from pandas import to_datetime
from utils import minutes_to_midnight

# GLOBAL MODEL PARAMETERS
alpha = 0.1
phi = 0.001
delta = 0

class DJSPEnv(gym.Env):
    """
    An operation scheduling environment for OpenAI gym, developed specifically for the scheduling of ground handling
    equipment to aircraft. Capable of considering parallel processes.
    """

    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, instance_path=None, a=None, p=None):
        # set global variables if provided
        if a:
            global alpha=a
        if p:
            global phi=p

        # declare variables which are required to store information from the instance specification
        self.n_aircraft = 0
        self.n_operations = 0
        self.machines_per_op = []
        self.n_machines = 0
        self.aircraft = []  # list of dictionaries containing relevant flight information
        self.parallel_mask = np.empty(shape=(1, 1), dtype=bool)

        # load instance and initialise relevant matrices
        if instance_path:
            self.load_instance(instance_path)

        # declare remaining variables
        self.assignment = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=bool)
        self.time_availability = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines),
                                          dtype=pd.Timestamp)
        # self.time_availability_discrete = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=int)
        self.availability = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=bool)
        self.operation_times = np.empty(shape=(self.n_aircraft, self.n_operations), dtype=dict)
        self.time_conflicts = np.empty(shape=(self.n_operations, self.n_aircraft, self.n_aircraft), dtype=bool)
        self.delays = {'Operative Delays': np.zeros(shape=(self.n_aircraft, self.n_operations)),
                       'Total Operative Delay': 0,
                       'Aircraft Delays': np.zeros(shape=self.n_aircraft),
                       'Total Aircraft Delay': 0}
        self.current_observation = {}
        self.max_delays = self.calculate_max_delays()
        self.rewards = {'per aircraft': np.zeros(shape=self.n_aircraft),
                        'total reward': 0}

        # fill matrices and set action mask
        self._init_availability()
        self._init_operation_times()
        self._init_timeavailability()
        self.action_mask = np.ravel(self.availability).astype(np.int8)

        # save empty matrices for easy environment resetting
        self.init_assignment = self.assignment
        self.init_availability = self.availability
        self.init_operation_times = self.operation_times
        self.init_time_availability = self.time_availability
        # self.init_time_availability_discrete = self.time_availability_discrete

        # initialise action and observation space
        self._init_actionspace()
        self._init_observationspace()

    def reset(self, **kwargs):
        """
        Returns the observation of the initial state and resets the environment to the initial state so that a new
        episode (independent of previous ones) may start.
        """
        # Reset assignment, availability and operation time matrices
        self.assignment = self.init_assignment
        self.availability = self.init_availability
        self.time_availability = self.init_time_availability
        # self.time_availability_discrete = self.init_time_availability_discrete
        self.operation_times = self.init_operation_times

        # reset observation space
        self.current_observation = {
            'assignment matrix': self.init_assignment,
            'total aircraft delay': self.delays['Aircraft Delays']
        }
        return self.current_observation

    def step(self, action):
        """
        Performs one step with the given action. Transforms the current observation, calculates the reward. Assignment
        update automatically performs the following updates as well:
         - availability matrices
         - operation times
         - action mask
         - aircraft delay
        """
        # update assignment, which automatically performs other updates (indices calculated in assignment update)
        aircraft_index = self.update_assignment(action)

        # update observation
        self.current_observation, new_delay = self._transform_observation()
        reward = self._calculate_reward(self.current_observation)

        # terminate only when no machine-operation assignments are available / feasible
        if self.time_availability.any():
            terminate = False
        else:
            terminate = True

        return self.current_observation, reward, terminate

    def sample_action(self):
        """ Samples action under consideration of machine availability (flattened and converted from bool to np.int8)"""
        return self.action_space.sample(mask=self.action_mask)

    def _transform_observation(self):
        """ Generates the new observation """
        new_observation = {
            'assignment matrix': self.assignment,
            'total aircraft delay': self.delays['Total Aircraft Delay']
        }
        return new_observation

    def _calculate_reward(self, delay, aircraft_index, function='linear'):
        """ Calculates the value of the reward function. """
        delay =
        # calculate the aircraft reward for the assigned
        if delay == 0:
            return self.rewards['total reward']
        elif delay < 15:
            self.rewards['per aircraft'][aircraft_index] = self.pre_delay_reward_function(delay)
        elif delay >= 15:
            if function == 'linear':
                self.rewards['per aircraft'][aircraft_index] = self.linear_reward(delay)
            elif function == 'exponential':
                self.rewards['per aircraft'][aircraft_index] = self.exponential_reward(delay)

    def pre_delay_reward_function(self, delay):
        return (alpha / 15) * delay

    def _linear_intercept(self):
        return 1 - ((1 - alpha) / (delta - 15)) * delta

    def linear_reward(self, delay):
        m = self._linear_intercept()
        return ((1 - alpha) / (delta - 15)) * delay + m

    def _exponental_intercept(self):
        r = alpha - (phi * np.e ** (15 - delta))
        return r

    def exponential_reward(self, delay):
        beta = self._exponental_intercept()
        return phi * np.exp(delay - delta) + beta

    def load_instance(self, instance_path):
        """
        Loads important problem information from the instance given at instance_path. The required format for instance
        specifications is given in the documentation.
        """
        with open(instance_path, 'r') as file:
            lines = file.readlines()
            file.close()

        # general information on number of aircraft, operation types, machines per operation and total machines
        self.n_aircraft, self.n_operations = [int(n) for n in lines[0].split()]
        self.machines_per_op = [int(n) for n in lines[1].split()]
        self.n_machines = np.sum(self.machines_per_op)

        # read information for each aircraft
        for linenr in range(2, 2 + self.n_aircraft):
            line = lines[linenr].split()
            info = {'REG': line[0],
                    'ETA': to_datetime(line[1], format='%H%M'),
                    'STD': to_datetime(line[2], format='%H%M'),
                    'Processing Times': [Timedelta(minutes=int(t)) for t in line[3:]]}
            self.aircraft.append(info)

        # matrix of tasks which can be run in parallel
        self.parallel_mask = np.empty((self.n_operations, self.n_operations), dtype=bool)
        s = 2 + self.n_aircraft
        for idx in range(s, s + self.n_operations):
            self.parallel_mask[idx - s] = [int(x) for x in lines[idx].split()]

    # INITIALISATION FUNCTIONS ----------------------------------------------------------------------------------------

    def _init_timeavailability(self):
        """
        Initialises the availability matrix containing earliest machine availability. This is the alternative to the
        binary availability matrix and allows for delayed assignment. Time is given in minutes since midnight.
        """
        type_start = 0
        for operationtype, nmachines in enumerate(self.machines_per_op):
            next_type_start = type_start + nmachines
            for aircraft in range(self.n_aircraft):
                # assign machine earliest availability to operation earliest start
                start = self.operation_times[aircraft, operationtype]['Earliest Start']
                self.time_availability[aircraft, operationtype, type_start:next_type_start] = start
                # self.time_availability_discrete[aircraft, operationtype,
                #                                 type_start:next_type_start] = minutes_since_midnight(start)
            type_start = next_type_start

    def _init_actionspace(self):
        """
        Initialises action space as a Discrete(dim). The dimension of the Discrete action space is a function of on the
        number of machines per operation type and the number of aircraft,
        where dim = sum_{i=1}^k(n_aircraft * n_machines_k)
        """
        # Initialise Discrete actionspace
        dim = 0
        for op_type in range(0, self.n_operations):
            dim += self.n_aircraft * self.machines_per_op[op_type] * self.n_operations
        self.action_space = Discrete(dim)

    def _init_observationspace(self):
        """
        Initialises the observation space as a dictionary of simple spaces. Currently implemented are:
          1. assignment matrix
          2. total aircraft delay
        """
        delaydim = 0
        self.observation_space = Dict(
            {
                'assignment matrix': MultiBinary([self.n_aircraft, self.n_operations, self.n_operations]),
                'total aircraft delay': Discrete(delaydim)
            }
        )

    def _init_availability(self):
        """
        Initialises the availability matrix showing which machines can be used for which aircraft operations, the matrix
        has the shape [aircraft x operations x machines].
        """
        type_start = 0
        for operation_type, nmachines in enumerate(self.machines_per_op):
            next_type_start = type_start + nmachines

            # assign availablity and machine upper and lower bounds
            self.availability[:, operation_type, type_start:next_type_start] = 1

            # assign variables for next iteration
            type_start = next_type_start

    def _init_operation_times(self):
        """
        Initialises the operation times dict-arrays containing earliest start & end and scheduled start & end times
        """
        for aircraft in range(0, self.n_aircraft):
            for operation in range(0, self.n_operations):
                # check that earliest times have not already been calculated by a successive operative
                if self.operation_times[aircraft, operation]:
                    continue
                start, end = self.earliest_times(operation, aircraft)
                self.operation_times[aircraft, operation] = {'Earliest Start': start, 'Earliest End': end,
                                                             'Scheduled Start': None, 'Scheduled End': None,
                                                             'Latest Start': None, 'Current Delay': 0}

        # use operation times to initialise time conflict matrix
        self._init_time_conflicts()

    def _init_time_conflicts(self):
        """
        Per operation, a [n_aircraft x n_aircraft]-matrix is created showing which operations from aircraft are in
        temporal conflict with one another. This is done per operation type, so conflict between operations of
        different types are not considered.
        :return:
        """
        for op in range(0, self.n_operations):
            for a1 in range(0, self.n_aircraft):
                for a2 in range(0, self.n_aircraft):
                    if self.time_conflicts[op, a1, a2]:
                        continue
                    elif a1 == a2:
                        self.time_conflicts[op, a1, a2] = 1
                    else:
                        x = 5
                        if self.operation_times[a1, op]['Earliest Start'] <= self.operation_times[a2, op][
                            'Earliest End'] and \
                                self.operation_times[a2, op]['Earliest Start'] <= self.operation_times[a1, op][
                            'Earliest End']:
                            self.time_conflicts[op, a1, a2] = 1
                            self.time_conflicts[op, a2, a1] = 1

    # UPDATE FUNCTIONS ------------------------------------------------------------------------------------------------

    def update_assignment(self, action):
        """
        Updates assignment for a given action and write scheduled time into operation_times.
        Machine availability is also adjusted.
        """
        # convert integer action to aircraft, operation and machine index, and set assignment to 1
        aircraft_index, operation_index, machine_index = self.convert_action_to_assignment(action)
        self.assignment[aircraft_index, operation_index, machine_index] = 1

        # perform other updates which require index information
        self.update_availability(aircraft_index, operation_index)
        self.update_operation_times(aircraft_index, operation_index, machine_index)
        self.update_time_availability(aircraft_index, operation_index, machine_index)
        self.update_action_mask()
        self.update_delay(aircraft_index, operation_index)

        return aircraft_index

    def update_availability(self, ac_index, op_index):
        """
        For a sampled action, the availability matrix is adjusted to show the operation at op_index as unavailable for
        assignment to all other machines.
        """
        self.availability[ac_index, op_index, :] = 0
        # conflict_aircraft_idxs = np.where(self.time_conflicts[op_index, ac_index, :])[0]
        # for aircraft_idx in conflict_aircraft_idxs:
        #     self.availability[aircraft_idx, op_index, mach_index] = 0

    def update_time_availability(self, ac_index, op_index, mach_index):
        """
        Updates the time availability matrices (both Timestamp and discrete time matrices). Also automatically updates
        the time conflict matrix.
        """
        # make assigned operation unavailable for all machines
        self.time_availability[ac_index, op_index, :] = 0
        # self.time_availability_discrete[ac_index, op_index, :] = 0

        # update time conflict first to consider new conflicts for delayed assignment
        self.update_time_conflicts(ac_index, op_index, mach_index)

        # set new earliest available time for operations which overlap in time with the assigned operation
        conflict_aircraft_idxs = np.where(self.time_conflicts[op_index, ac_index, :])[0]
        for aircraft_idx in conflict_aircraft_idxs:
            self.time_availability[aircraft_idx, op_index, mach_index] = \
                self.operation_times[aircraft_idx, op_index, mach_index]['Scheduled End']
            # self.time_availability_discrete[aircraft_idx, op_index, mach_index] = \
            #     minutes_since_midnight(self.operation_times[aircraft_idx, op_index, mach_index]['Scheduled End'])

    def update_operation_times(self, ac_index, op_index, mach_index):
        """
        Updates the scheduled start and end times.
        """
        start = self.time_availability[ac_index, op_index, mach_index]
        self.operation_times[ac_index, op_index]['Scheduled Start'] = start
        self.operation_times[ac_index, op_index]['Scheduled End'] = start + self.aircraft[ac_index]['Processing Times'][
            op_index]

    def update_time_conflicts(self, ac_index, op_index, mach_index):
        """
        Updates conflict matrix for a given action (aircraft, operation and machine indices)
        """
        # only check for aircraft whose operation at op_index can be assigned to this machine (gives aircraft indices)
        feasible_assignments = np.where(self.availability[:, op_index, mach_index])

        for ac in feasible_assignments:
            if ac == ac_index:
                continue
            # if time overlaps, ensure time conflict = 1 unless
            if self.operation_times[ac, op_index]['Earliest Start'] <= self.operation_times[ac_index, op_index][
                'Scheduled End'] and \
                    self.operation_times[ac, op_index]['Earliest Start'] <= self.operation_times[ac_index, op_index][
                'Scheduled End'] and \
                    not self.time_conflicts[op_index, ac, ac_index] and not self.time_conflicts[op_index, ac_index, ac]:
                self.time_conflicts[op_index, ac, ac_index] = 1
                self.time_conflicts[op_index, ac_index, ac] = 1
            else:
                if self.time_conflicts[op_index, ac, ac_index] or self.time_conflicts[op_index, ac_index, ac]:
                    self.time_conflicts[op_index, ac, ac_index] = 0
                    self.time_conflicts[op_index, ac_index, ac] = 0

    def update_action_mask(self):
        """
        Updates the action mask with the new availability matrix --> flattened and converted from bool to np.int8 as
        required by the .sample() function
        """
        self.action_mask = np.ravel(self.availability).astype(np.int8)

    def update_timeavailability(self, ac_index, op_index, mach_index):
        """
        Updates the timeavailability matrix cfor the machine at mach_index, setting it to 0 for the assigned operation
        at op_index and adjusting the earliest times for all other operations.
        """
        # determine new earliest availability for machine at mach_index
        self.time_availability[ac_index, op_index, mach_index] = 0
        for aircraft in range(self.n_aircraft):
            self.time_availability[aircraft, op_index, mach_index] = self.operation_times[ac_index, op_index][
                'Scheduled End']

    def update_delay(self, ac_index, op_index):
        """
        Updates delay dictionary, adding delay in minutes to list of delays, adding to total delay and recalculating
        the average delay over all delayed operations
        """
        delay = int(self.operation_times[ac_index, op_index]['Scheduled End'] - \
                    self.operation_times[ac_index, op_index]['Earliest End'])
        if delay != 0:
            # Add delay to operative delays list and total operative delay
            self.delays['Operative Delays'][ac_index, op_index] = delay
            self.delays['Total Operative Delay'] += delay

            # update delay for following operations
            self.update_following_delays(ac_index, op_index)
            self.update_aircraft_delay(ac_index, op_index)

    def update_following_delays(self, ac_index, op_index):
        """
        Updates the delay for all operations following the operation at op_index for the aircraft at ac_index
        """
        following_ops = self.following(op_index)
        for op in following_ops:
            if self.operation_times[ac_index, op]['Scheduled Start']:
                self.operation_times[ac_index, op_index]['Current Delay'] =

    def update_aircraft_delay(self, ac_index, op_index):
        """
        Identifies whether aircraft is delayed by the delayed assignment of an operation.
        """

        dep = self.aircraft[ac_index]['STD']
        if not following_ops:
            # evaluate if the delayed operation causes aircraft delay
            if self.operation_times[ac_index, ]['Scheduled End']
        else:
    # evaluate for foll


    # UTILITY FUNCTIONS -----------------------------------------------------------------------------------------------

    def convert_action_to_assignment(self, action_index):
        """
        Action sampled from the action space is converted to an operation assignment. Returns the index of the
        assignment cell the action corresponds to. Used by the update_assignment function
        """
        aircraft_index = int(action_index / (self.n_operations * self.n_machines))
        normalised_index = action_index - (self.n_operations * self.n_machines * aircraft_index)
        operation_index = int(normalised_index / self.n_machines)
        machine_index = normalised_index - (self.n_machines * operation_index)

        return aircraft_index, operation_index, machine_index

    def earliest_times(self, op_idx, ac_idx):
        """
        Gathers earliest start from precedence function and uses it to calculate the earliest end time for an operation
        """
        earliest_start = self.precedence(op_idx, ac_idx)
        earliest_end = earliest_start + self.aircraft[ac_idx]['Processing Times'][op_idx]
        return earliest_start, earliest_end

    def precedence(self, op_idx, ac_idx):
        """
        Identifies operations preceding op_idx and returns the earliest end time of that operation, which is also the
        earliest start time of the successive operation.
        """
        # identify potential preceding operations - where the column [:,op] for operation is True, return None if empty
        precedence_idxs = np.where(self.parallel_mask[:, op_idx])[0]

        # aircraft ETA is the earliest start time with no preceding ops
        if not precedence_idxs.size:
            return self.aircraft[ac_idx]['ETA']
        else:
            # check that the operation is not a parallel task (parallel when [op,idx]==1 and [idx,op]==1)
            earliest_start = self.aircraft[ac_idx]['ETA']
            for preceding_op in precedence_idxs:
                if not self.parallel_mask[op_idx, preceding_op]:
                    # if not self.operation_times[ac_idx, op_idx]:
                    #     # calculate earliest start and end times for the preceding operation
                    #     prec_start, prec_end = self.earliest_times(preceding_op, ac_idx)
                    #     self.operation_times[ac_idx, preceding_op] = {'Earliest Start': prec_end, 'Earliest End': prec_end,
                    #                                          'Scheduled Start': None, 'Scheduled End': None}
                    earliest_start = self.operation_times[ac_idx, preceding_op]['Earliest End']

            return earliest_start

    def following(self, op_idx):
        # identify potential following operations - where the column
        following_idxs = np.where(self.parallel_mask[op_idx, :])[0]
        following_ops = []

        if not following_idxs.size:
            return None
        for following_op in following_idxs:
            if not self.parallel_mask[following_op, op_idx]:
                following_ops.append(following_op)

        return following_ops

    def calculate_max_delays(self):
        """
        Calculates maximum possible delay delta for each aircraft in the schedule delta,
        used to define the reward function uniquely for each aircraft
        """
        delta = []
        for ac in self.aircraft:
            delta.append(minutes_to_midnight(self.aircraft[ac]['STD']))
        return delta
