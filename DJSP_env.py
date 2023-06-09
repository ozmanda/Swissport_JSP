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
            global alpha
            alpha = a
        if p:
            global phi
            phi = p

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
            self.dim = self.n_aircraft * self.n_operations * self.n_machines

        # declare remaining variables
        self.assignment = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=np.uint8)
        self.time_availability = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines),
                                          dtype=pd.Timestamp)
        # self.time_availability_discrete = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=int)
        self.availability = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=bool)
        self.operation_pointers = np.zeros(shape=(self.n_operations, 2), dtype=list)
        self.operation_times = np.empty(shape=(self.n_aircraft, self.n_operations), dtype=dict)
        # self.time_conflicts = np.empty(shape=(self.n_operations, self.n_aircraft, self.n_aircraft), dtype=bool)
        self.delays = {'Operative Delays': np.zeros(shape=(self.n_aircraft, self.n_operations)),
                       'Total Operative Delay': 0,
                       'Aircraft Delays': np.zeros(shape=self.n_aircraft),
                       'Total Aircraft Delay': 0}
        self.current_observation = MultiBinary([self.n_aircraft, self.n_operations, self.n_machines])
        self.max_delays = self.calculate_max_delays()
        self.delaydim = np.sum(self.max_delays)
        self.rewards = {'per aircraft': np.zeros(shape=self.n_aircraft),
                        'total reward': 0}
        self.steps = 0

        # fill matrices and set action mask
        self._init_operation_pointers()
        self._init_availability()
        self._init_operation_times()
        self._init_timeavailability()
        self.action_mask = np.ravel(self.availability).astype(np.int8)
        self.last_operations = np.where(self.operation_pointers[:, 1] == None)[0]

        # save empty matrices for easy environment resetting
        self.init_assignment = self.assignment.copy()
        self.init_availability = self.availability.copy()
        self.init_operation_times = self.operation_times.copy()
        self.init_time_availability = self.time_availability.copy()
        self.init_delays = self.delays.copy()
        self.init_action_mask = self.action_mask.copy()
        # self.init_time_availability_discrete = self.time_availability_discrete

        # initialise action and observation space
        self._init_actionspace()
        self._init_observationspace()

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

        # load processing time information
        processing_times = [Timedelta(minutes=int(t)) for t in lines[2].split()]

        # matrix of tasks which can be run in parallel
        self.parallel_mask = np.empty((self.n_operations, self.n_operations), dtype=bool)
        for idx in range(3, 3+self.n_operations):
            self.parallel_mask[idx-3] = [int(x) for x in lines[idx].split()]

        # read information for each aircraft
        for linenr in range(3+self.n_operations, len(lines)):
            line = lines[linenr].split()
            info = {'REG': line[0],
                    'ETA': to_datetime(line[1], format='%H%M'),
                    'STD': to_datetime(line[2], format='%H%M'),
                    'Processing Times': processing_times}
            self.aircraft.append(info)


    # CORE FUNCTIONS --------------------------------------------------------------------------------------------------

    def reset(self, **kwargs):
        """
        Returns the observation of the initial state and resets the environment to the initial state so that a new
        episode (independent of previous ones) may start.
        """
        # Reset assignment, availability and operation time matrices
        self.assignment = np.array(self.init_assignment, dtype=np.uint8).copy()
        self.availability = self.init_availability.copy()
        self.time_availability = self.init_time_availability.copy()
        # self.time_availability_discrete = self.init_time_availability_discrete
        self.operation_times = self.init_operation_times.copy()
        self.delays = self.init_delays.copy()

        # reset observation space
        # self.current_observation = np.array(self.assignment, dtype=np.uint8)
        self.current_observation = np.zeros(shape=(2, self.dim))
        self.action_mask = self.init_action_mask.copy()
        self.current_observation[1, :] = self.action_mask
        self.rewards = {'per aircraft': np.zeros(shape=self.n_aircraft),
                        'total reward': 0}
        self.steps = 0
        # self.current_observation = {
        #     'assignment matrix': self.init_assignment,
        #     'total aircraft delay': self.delays['Aircraft Delays']
        # }
        return self.current_observation, 0

    def step(self, action):
        """
        Performs one step with the given action. Transforms the current observation, calculates the reward. Assignment
        update automatically performs the following updates as well:
         - availability matrices
         - operation times
         - action mask
         - aircraft delay
        """
        self.steps += 1
        if self.action_mask[action]:
            # update assignment, which automatically performs other updates (indices calculated in assignment update)
            aircraft_index = self.perform_updates(action)

            # update observation
            self.current_observation = self._transform_observation()
            reward = self._calculate_reward(aircraft_index)
        else:
            reward = self._calculate_reward(0, invalid=True)

        truncated = True if self.steps >= self.n_aircraft * self.n_operations * 100000 else False
        terminated = False if self.availability.any() else True
        self.steps += 1

        # if not self.steps%20:
        #     if self.assignment.any():
        #         print(f'{self.assignment}\n\n\n')

        return self.current_observation, reward, terminated, truncated, {}

    def sample_action(self):
        """ Samples action under consideration of machine availability (flattened and converted from bool to np.int8)"""
        return self.action_space.sample(mask=self.action_mask)

    def _transform_observation(self):
        """ Generates the new observation """
        obs = np.zeros(shape=(2, self.dim), dtype=np.uint8)
        obs[0, :] = np.ravel(self.assignment)
        obs[1, :] = self.action_mask
        self.current_observation = obs
        # self.current_observation = np.array(self.assignment, dtype=np.uint8)
        # self.current_observation['total aircraft delay'] = Discrete(self.delays['Total Aircraft Delay'].astype(np.uint8))
        # new_observation = {
        #     'assignment matrix': np.array(self.assignment, dtype=int),
        #     'total aircraft delay': int(self.delays['Total Aircraft Delay'])
        # }
        # self.current_observation = new_observation
        return self.current_observation
        # return new_observation

    def _calculate_reward(self, aircraft_index, function='sparse', invalid=False):
        """ Calculates the value of the reward function. """
        reward = None
        if invalid:
            if function == 'sparse':
                reward = 0
            elif function == 'linear':
                reward = self.linear_reward(self.delaydim)
            elif function == 'exponential':
                reward = self.exponential_reward(self.delaydim)
            return reward
        else:
            delay = self.delays['Aircraft Delays'][aircraft_index]
            # calculate the aircraft reward for the assigned
            if delay == 0:
                reward = 1
            elif delay < 15:
                if function == 'sparse':
                    reward = 0.5
                elif function == 'linear' or function == 'exponential':
                    reward = self.pre_delay_reward_function(delay)
            elif delay >= 15:
                if function == 'sparse':
                    reward = 0
                elif function == 'linear':
                    reward = self.linear_reward(delay)
                elif function == 'exponential':
                    reward = self.exponential_reward(delay)

            self.rewards['per aircraft'][aircraft_index] = reward
            self.rewards['total reward'] = np.sum(self.rewards['per aircraft'])

            return reward

    def pre_delay_reward_function(self, delay):
        return (alpha / 15) * delay

    # CHECK IF INTERCEPTS ARE STATIC!!!!

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

    # INITIALISATION FUNCTIONS ----------------------------------------------------------------------------------------

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
        delaydim = np.sum(self.max_delays)
        self.observation_space = MultiBinary([2, self.dim])
        # self.observation_space = Dict(
        #     {
        #         'assignment matrix': MultiBinary([self.n_aircraft, self.n_operations, self.n_machines]),
        #         'total aircraft delay': Discrete(delaydim)
        #     }
        # )

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
                if not start:
                    raise TypeError
                self.time_availability[aircraft, operationtype, type_start:next_type_start] = start
                # self.time_availability_discrete[aircraft, operationtype,
                #                                 type_start:next_type_start] = minutes_since_midnight(start)
            type_start = next_type_start

    def _init_operation_pointers(self):
        """
        Initialises a "pointer" array of shape (n_ops, 2) containing the indices of operations preceding and following
        operations
        """
        for op_index in range(self.n_operations):
            self.operation_pointers[op_index, 0] = self.before(op_index)
            self.operation_pointers[op_index, 1] = self.after(op_index)

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
                                                             'Latest Start': None, 'Delayed Start': None,
                                                             'Current Delay': 0}

    # UPDATE FUNCTIONS ------------------------------------------------------------------------------------------------

    def perform_updates(self, action):
        """
        Performs all necessary updates which occur when an assignment is made:
          - sets assignment and scheduled start/end times for the operation
          - updates availability matrix for the assigned operation
          - updates latest/earliest start times for prior/following operations in case of delayed assignment
          - updates time conflicts in case of delayed assignment
          - updates machine earliest time availability for conflicting operations
        """
        # convert integer action to aircraft, operation and machine index, set assignment and calculate delay
        aircraft_index, operation_index, machine_index = self.convert_action_to_assignment(action)
        delay = self.update_assignment(aircraft_index, operation_index, machine_index)

        # remove operation from availability matrix
        self.availability[aircraft_index, operation_index, :] = 0
        self.time_availability[aircraft_index, operation_index, :] = 0

        # if operation is delayed, update time conflicts and set latest/earliest start times for prior/following ops
        if delay:
            self.update_prior_ops(operation_index, aircraft_index)
            self.update_following_ops(operation_index, aircraft_index, delay)
            # self.update_time_conflicts(aircraft_index, operation_index, machine_index)
            self.update_delay(aircraft_index, operation_index, delay)

        # update time availability for conflicting operations and action mask for next action selection
        self.update_time_availability(aircraft_index, operation_index, machine_index)
        self.update_action_mask()

        return aircraft_index

    def update_assignment(self, ac_idx, op_idx, mach_idx):
        """
        Updates assignment for a given action and write scheduled time into operation_times.
        """
        # set assignment, update operation times and determine if the assignment contains a delay
        self.assignment[ac_idx, op_idx, mach_idx] = 1
        delay = self.update_scheduled_times(ac_idx, op_idx, mach_idx)
        return delay

    def update_time_availability(self, ac_idx, op_idx, mach_idx):
        """
        Updates the time availability matrix.
        """
        # Identify aircraft with earliest time availabilty for the operation at op_idx at the machine at mach_idx
        ac_idxs = np.where(self.time_availability[:, op_idx, mach_idx])[0]
        scheduled_end = self.operation_times[ac_idx, op_idx]['Scheduled End']
        scheduled_start = self.operation_times[ac_idx, op_idx]['Scheduled Start']

        for ac in ac_idxs:
            if self.time_availability[ac, op_idx, mach_idx] <= scheduled_end and self.time_availability[
                ac, op_idx, mach_idx] + self.aircraft[ac]['Processing Times'][op_idx] >= scheduled_start:
                if self.operation_times[ac, op_idx]['Latest Start']:
                    if self.operation_times[ac, op_idx]['Latest Start'] < scheduled_end:
                        self.time_availability[ac, op_idx, mach_idx] = 0
                        self.availability[ac, op_idx, mach_idx] = 0
                    else:
                        self.time_availability[ac, op_idx, mach_idx] = scheduled_end
                else:
                    self.time_availability[ac, op_idx, mach_idx] = scheduled_end

        # adjust time availabiltiy of dependent, prior aircraft
        prior_ops = np.where(self.parallel_mask[:, op_idx])[0]
        for prior_op in prior_ops:
            mach_idxs = np.where(self.availability[ac_idx, op_idx, :])[0]
            for m_idx in mach_idxs:
                if m_idx == mach_idx:
                    continue
                else:
                    self.time_availability[ac_idx, prior_op, ]

    def update_scheduled_times(self, ac_index, op_index, mach_index):
        """
        Updates the scheduled start and end times and returns the delay
        """
        start = self.time_availability[ac_index, op_index, mach_index]
        if start == 0:
            x = 5
        self.operation_times[ac_index, op_index]['Scheduled Start'] = start
        try:
            self.operation_times[ac_index, op_index]['Scheduled End'] = start + self.aircraft[ac_index]['Processing Times'][
                op_index]
        except TypeError:
            x = 5
        delay_td = self.operation_times[ac_index, op_index]['Scheduled End'] - self.operation_times[ac_index, op_index][
            'Earliest End']
        return int(delay_td.total_seconds() / 60)

    def update_prior_ops(self, op_idx, ac_idx):
        """
        When an operation is assigned with delay, the latest start of all prior operations is updated, until the first
        operation in the sequence.
        """
        self.operation_times[ac_idx, op_idx]['Latest Start'] = self.operation_times[ac_idx, op_idx]['Scheduled Start']
        prior_op_idxs = {op_idx: self.operation_pointers[op_idx, 0]} if self.operation_pointers[op_idx, 0] else {}

        # iterate as long as there are still prior operations to be processed
        while prior_op_idxs:
            # empty dictionary to save next prior ops
            next_prior_op_idxs = {}

            # per operation, iterate through all prior operations
            for key in prior_op_idxs.keys():
                for prior_op_idx in prior_op_idxs[key]:
                    # check that the op is unassigned, otherwise prior ops are determined by this already assigned op
                    if not self.operation_times[ac_idx, prior_op_idx]['Scheduled Start']:
                        latest_start = self.operation_times[ac_idx, key]['Latest Start'] - \
                                       self.aircraft[ac_idx]['Processing Times'][prior_op_idx]
                        self.operation_times[ac_idx, prior_op_idx]['Latest Start'] = latest_start

                        # check for machines that are only available after the latest start time and remove them
                        mach_idxs = np.where(self.availability[ac_idx, prior_op_idx, :])[0]
                        for mach_idx in mach_idxs:
                            if self.time_availability[ac_idx, prior_op_idx, mach_idx] > latest_start:
                                self.time_availability[ac_idx, prior_op_idx, mach_idx] = 0
                                self.availability[ac_idx, prior_op_idx, mach_idx] = 0

                        # if the operation also has prior operations, append to the dictionary list for next iteration
                        if self.operation_pointers[prior_op_idx, 0]:
                            next_prior_op_idxs[prior_op_idx] = self.operation_pointers[prior_op_idx, 0]

            # update prior operation index list
            prior_op_idxs = next_prior_op_idxs

    def update_following_ops(self, op_idx, ac_idx, delay):
        """
        When an operation is assigned with delay, the earliest start of all following operations is updated, until the
        last operation in the sequence. For the operations following a delayed assignment, the current delay is saved.
        """
        self.operation_times[ac_idx, op_idx]['Delayed Start'] = self.operation_times[ac_idx, op_idx]['Scheduled Start']
        following_op_idxs = {op_idx: self.operation_pointers[op_idx, 1]} if self.operation_pointers[op_idx, 1] else {}

        # iterate as long as there are still following operations to be processed
        while following_op_idxs:
            # empty dictionary to save next following ops
            next_following_ops = {}

            # per operation, iterate through all following operations
            try:
                for prior_op in following_op_idxs.keys():
                    # calculate earliest end of the prior operation
                    earliest_start = self.operation_times[ac_idx, prior_op]['Delayed Start'] + \
                                     self.aircraft[ac_idx]['Processing Times'][prior_op]

                    # iterate through all operations dependant on the prior op
                    for following_op in following_op_idxs[prior_op]:
                        # check that the following op is unassigned, otherwise all ops after are already determined
                        if not self.operation_times[ac_idx, following_op]['Scheduled Start']:
                            self.operation_times[ac_idx, following_op]['Delayed Start'] = earliest_start
                            self.operation_times[ac_idx, following_op]['Current Delay'] = delay

                            # adjust time availability for available machines
                            available_machines = np.where(self.time_availability[ac_idx, following_op, :])[0]
                            for mach in available_machines:
                                if self.time_availability[ac_idx, following_op, mach] < earliest_start:
                                    self.time_availability[ac_idx, following_op, mach] = earliest_start

                            # if the operation also has following operations, append to the dict list for next iteration
                            if self.operation_pointers[following_op, 1]:
                                next_following_ops[following_op] = self.operation_pointers[following_op, 1]

            # update following operation index list
            following_op_idxs = next_following_ops

    def update_action_mask(self):
        """
        Updates the action mask with the new availability matrix --> flattened and converted from bool to np.int8 as
        required by the .sample() function
        """
        self.action_mask = np.ravel(self.availability).astype(np.int8)
        return self.action_mask

    def update_delay(self, ac_index, op_index, delay):
        """
        Updates delay dictionary, adding delay in minutes to list of delays, adding to total delay and recalculating
        the average delay over all delayed operations
        """
        # Add delay to operative delays list and total operative delay
        self.delays['Operative Delays'][ac_index, op_index] = delay
        self.delays['Total Operative Delay'] += delay

        # evaluate aircraft delay
        ac_delay = self.evaluate_aircraft_delay(ac_index)
        self.delays['Aircraft Delays'][ac_index] = ac_delay
        self.delays['Total Aircraft Delay'] = np.sum(self.delays['Aircraft Delays'])

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
        # if no prior operation must be completed, set earliest start to aircraft ETA
        earliest_start = self.aircraft[ac_idx]['ETA']
        if self.operation_pointers[op_idx, 0]:
            # gather earliest end of prior operation, set earliest start = the latest end of prior operations
            for idx in self.operation_pointers[op_idx, 0]:
                prior_end = self.operation_times[ac_idx, idx]['Earliest End']
                if prior_end > earliest_start:
                    earliest_start = prior_end

        earliest_end = earliest_start + self.aircraft[ac_idx]['Processing Times'][op_idx]
        return earliest_start, earliest_end

    def before(self, op_idx):
        """
        Identifies the index of the operation(s) which must be completed before the operation at op_index can begin
        and returns it/them in the form of a list
        """
        idxs = []
        precedence_idxs = np.where(self.parallel_mask[:, op_idx])[0]

        # iterate through potential preceding operations and append operation indices which are not parallel
        if not precedence_idxs.size:
            return None
        else:
            for op in precedence_idxs:
                if not self.parallel_mask[op_idx, op]:
                    idxs.append(op)

        if not len(idxs):
            return None
        else:
            return idxs

    def after(self, op_idx):
        """
        Identifies the index of the operation(s) which must be completed after the operation at op_index can begin
        and returns it/them in the form of a list
        """
        idxs = []
        following_idxs = np.where(self.parallel_mask[op_idx, :])[0]

        if not following_idxs.size:
            return None
        else:
            for op in following_idxs:
                if not self.parallel_mask[op, op_idx]:
                    idxs.append(op)

        if not len(idxs):
            return None
        else:
            return idxs

    def calculate_max_delays(self):
        """
        Calculates maximum possible delay delta for each aircraft in the schedule, used to define the reward
        function uniquely for each aircraft.
        """
        delta = []
        for ac in range(self.n_aircraft):
            delta.append(minutes_to_midnight(self.aircraft[ac]['STD']))
        return delta

    def evaluate_aircraft_delay(self, ac):
        std = self.aircraft[ac]['STD']
        latest_end = std
        for op in self.last_operations:
            # If the operation is scheduled, take the scheduled time, otherwise calculate the current earliest end
            if self.operation_times[ac, op]['Scheduled End']:
                end = self.operation_times[ac, op]['Scheduled End']
            else:
                end = self.operation_times[ac, op]['Delayed Start'] + self.aircraft[ac]['Processing Times'][op]

            # if end is after the current latest end, save the time
            if end > latest_end:
                latest_end = end

        return int((latest_end - std).total_seconds() / 60)

    def evaluate_schedule(self):
        """
        Checks the schedule for errors
        """
        # for each machine, check that the assigned operations are not in conflict with one another
        for machine in range(self.n_machines):
            operation = np.where(self.assignment)
