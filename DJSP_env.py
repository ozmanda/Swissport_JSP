import numpy as np
from pandas import to_datetime, Timedelta
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete
from pandas import to_datetime



class DJSPEnv(gym.Env):
    """
    An operation scheduling environment for OpenAI gym, developed specifically for the scheduling of ground handling
    equipment to aircraft. Capable of considering parallel processes.
    """
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, instance_path=None, render_mode=None):
        self.n_aircraft = 0
        self.n_operations = 0
        self.machines_per_op = []
        self.n_machines = 0
        self.aircraft = []  # list of dictionaries containing relevant flight information
        self.parallel_mask = np.empty(shape=(1, 1), dtype=bool)
        self.assignment = np.zeros(shape=(1, 1), dtype=bool)
        self.availability = np.zeros(shape=(1, 1), dtype=bool)
        self.operation_times = np.empty(shape=(1, 1), dtype=dict)
        self.time_conflicts = np.empty(shape=(1, 1), dtype=bool)
        self.current_observation = {}

        # load instance and initialise relevant matrices
        if instance_path:
            self.load_instance(instance_path)
        self._init_availability()
        self.action_mask = np.ravel(self.availability).astype(np.int8)
        self._init_assignment()
        self._init_operation_times()

        # save empty matrices for easy environment resetting
        self.init_assignment = self.assignment
        self.init_availability = self.availability
        self.init_operation_times = self.operation_times

        # initialise action and observation space
        self._init_actionspace()
        self._init_observationspace()

    def reset(self):
        """
        Returns the observation of the initial state and resets the environment to the initial state so that a new
        episode (independent of previous ones) may start.
        """
        # Reset assignment, availability and operation time matrices
        self.assignment = self.init_assignment
        self.availability = self.init_availability
        self.operation_times = self.init_operation_times

        # reset observation space
        self.current_observation = {
            'assignment matrix': self.init_assignment,
            'availability matrix': self.init_availability,
            'unassigned operations': self.n_operations * self.n_aircraft
        }
        return self.current_observation

    def step(self, action):
        """
        Performs one step with the given action. Transforms the current observation, calculates the reward.
        """
        # update assignment, which automatically updates availability and operation times
        self.update_assignment(action)

        # update observation
        self.current_observation = self._transform_observation()
        reward = self._calculate_reward(self.current_observation)

        if self.current_observation['unassigned operations'] == 0:
            terminate = True
        else:
            terminate = False

        return self.current_observation, reward, terminate

    def sample_action(self):
        """
        Samples action under consideration of machine availability (flattened and converted from bool to np.int8)
        """
        return self.action_space.sample(mask=self.action_mask)

    def _transform_observation(self):
        """
        Generates the new observation
        """
        new_observation = {
            'assignment matrix': self.assignment,
            'availability matrix': self.availability,
            'unassigned operations': self.current_observation['unassigned operations'] - 1
        }
        return new_observation

    def _calculate_reward(self, new_observation):
        """
        Calculates the value of the reward function.
        """
        if self.current_observation['unassigned operations'] > new_observation['unassigned operations']:
            reward = 1
        else:
            reward = 0

        return reward

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
        for linenr in range(2, 2+self.n_aircraft):
            line = lines[linenr].split()
            info = {'REG': line[0],
                    'ETA': to_datetime(line[1], format='%H%M'),
                    'STD': to_datetime(line[2], format='%H%M'),
                    'Processing Times': [int(t) for t in line[3:]]}
            self.aircraft.append(info)

        # matrix of tasks which can be run in parallel
        self.parallel_mask = np.empty((self.n_operations, self.n_operations), dtype=bool)
        s = 2+self.n_aircraft
        for idx in range(s, s+self.n_operations):
            self.parallel_mask[idx-s] = [int(x) for x in lines[idx].split()]

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

        IDEAS:
         - assignment matrix
         - availability matrix
         - number of unassigned operations
         - number of tardy flights
         - total tardiness [min]
         - utilisation rate of each machine [%]
        """
        self.observation_space = Dict(
            {
                'assignment matrix': MultiBinary([self.n_aircraft, self.n_operations, self.n_operations]),
                'availability matrix': MultiBinary([self.n_aircraft, self.n_operations, self.n_operations]),
                'unassigned operations': Discrete(self.n_operations)
            }
        )

    def _init_availability(self):
        """
        Initialises the availability matrix showing which machines can be used for which aircraft operations, the matrix
        has the shape [aircraft x operations x machines].
        """
        self.availability = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=bool)
        type_start = 0
        for operation_type, nmachines in enumerate(self.machines_per_op):
            next_type_start = type_start+nmachines

            # assign availablity and machine upper and lower bounds
            self.availability[:, operation_type, type_start:next_type_start] = 1

            # assign variables for next iteration
            type_start = next_type_start

    def _init_assignment(self):
        """
        Initialises the assignment matrix [aircraft x operations x machines] for each type of operation
        :return:
        """
        self.assignment = np.zeros(shape=(self.n_aircraft, self.n_operations, self.n_machines), dtype=bool)

    def _init_operation_times(self):
        """
        Initialises the operation times dict-arrays containing earliest start & end and scheduled start & end times
        """
        self.operation_times = np.empty(shape=(self.n_aircraft, self.n_operations), dtype=dict)
        for aircraft in range(0, self.n_aircraft):
            for operation in range(0, self.n_operations):
                # check that earliest times have not already been calculated by a successive operative
                if self.operation_times[aircraft, operation]:
                    continue
                start, end = self.earliest_times(operation, aircraft)
                self.operation_times[aircraft, operation] = {'Earliest Start': start, 'Earliest End': end,
                                                             'Scheduled Start': None, 'Scheduled End': None}

        # use operation times to initialise time conflict matrix
        self._init_time_conflicts()

    def _init_time_conflicts(self):
        """
        Per operation, a [n_aircraft x n_aircraft]-matrix is created showing which operations from aircraft are in
        temporal conflict with one another. This is done per operation type, so conflict between operations of
        different types are not considered.
        :return:
        """
        self.time_conflicts = np.zeros(shape=(self.n_operations, self.n_aircraft, self.n_aircraft), dtype=bool)
        for op in range(0, self.n_operations):
            for a1 in range(0, self.n_aircraft):
                for a2 in range(0, self.n_aircraft):
                    if a1 == a2:
                        self.time_conflicts[op, a1, a2] = 1
                    else:
                        if self.operation_times[a1, op]['Earliest Start'] >= self.operation_times[a2, op]['Earliest End'] and \
                           self.operation_times[a2, op]['Earliest Start'] <= self.operation_times[a1, op]['Earliest End']:
                            self.time_conflicts[op, a1, a2] = 1

    def update_assignment(self, action):
        """
        Updates assignment for a given action and write scheduled time into operation_times.
        Machine availability is also adjusted.
        """
        aircraft_index, operation_index, machine_index = self.convert_action_to_assignment(action)
        self.assignment[aircraft_index, operation_index, machine_index] = 1
        self.update_availability(aircraft_index, operation_index, machine_index)
        self.update_action_mask()
        self.update_operation_times(aircraft_index, operation_index)

    def update_availability(self, ac_index, op_index, mach_index):
        """
        For a sampled action, the availability matrix is adjusted to show the machine at mach_index as unavailable for
        all aircraft whose operation op_index coincides with the operation chosen.
        """
        self.availability[ac_index, op_index, :] = 0
        conflict_aircraft_idxs = np.where(self.time_conflicts[op_index, ac_index, :])[0]
        for aircraft_idx in conflict_aircraft_idxs:
            self.availability[aircraft_idx, op_index, mach_index] = 0

    def update_operation_times(self, ac_index, op_index):
        """
        Updates the scheduled start and end times.
        """
        self.operation_times[ac_index, op_index]['Scheduled Start'] = self.operation_times[ac_index, op_index]['Earliest Start']
        self.operation_times[ac_index, op_index]['Scheduled End'] = self.operation_times[ac_index, op_index]['Earliest End']

    def update_action_mask(self):
        """
        Updates the action mask with the new availability matrix --> flattened and converted from bool to np.int8 as
        required by the .sample() function
        """
        self.action_mask = np.ravel(self.availability).astype(np.int8)

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
        earliest_end = earliest_start + Timedelta(minutes=self.aircraft[ac_idx]['Processing Times'][op_idx])
        return earliest_start, earliest_end

    def precedence(self, op_idx, ac_idx):
        """
        Identifies operations preceding op_idx and returns the earliest end time of that operation, which is also the
        earliest start time of the successive operation.
        """
        # identify potential preceding operations - where the column [:,op] for operation is True, return None if empty
        precedence_idxs = np.where(self.parallel_mask[:, op_idx])[0]

        # aircraft ETA is the earliest start time with no preceding ops
        if precedence_idxs.size == 0:
            return self.aircraft[ac_idx]['ETA']

        else:
            # check that the operation is not a parallel task (parallel when [op,idx]==1 and [idx,op]==1)
            earliest_start = self.aircraft[ac_idx]['ETA']
            for preceding_op in precedence_idxs:
                if not self.parallel_mask[op_idx, preceding_op]:
                    if not self.operation_times[ac_idx, op_idx]:
                        # calculate earliest start and end times for the preceding operation
                        prec_start, prec_end = self.earliest_times(preceding_op, ac_idx)
                        self.operation_times[ac_idx, preceding_op] = {'Earliest Start': prec_start, 'Earliest End': prec_end,
                                                             'Scheduled Start': None, 'Scheduled End': None}
                    earliest_start = self.operation_times[ac_idx, preceding_op]['Earliest Start']

            return earliest_start