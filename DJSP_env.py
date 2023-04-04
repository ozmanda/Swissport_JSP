import numpy as np
from pandas import to_datetime, Timedelta
import gymnasium as gym
from gymnasium import spaces
from pandas import to_datetime


class DJSPEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, instance_path=None):
        self.n_aircraft = 0
        self.n_operations = 0
        self.machines_per_op = []
        self.n_machines = 0
        self.aircraft = []  # list of dictionaries containing relevant flight information
        self.parallel_mask = np.empty(shape=(1, 1), dtype=bool)
        self.assignment = []
        self.availability = []
        self.operation_times = []

        # load instance and initialise relevant matrices
        if instance_path:
            self.load_instance(instance_path)
        self._init_availability()
        self._init_assignment()
        self._init_operation_times()

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

        # read information for each aircraft
        for linenr in range(2, 2+self.n_aircraft):
            line = lines[linenr].split()
            info = {'ETA': to_datetime(line[0], format='%H%M').time(),
                    'STD': to_datetime(line[1], format='%H%M').time(),
                    'Processing Times': [int(t) for t in line[2:]]}
            self.aircraft.append(info)

        # matrix of tasks which can be run in parallel
        self.parallel_mask = np.empty((self.n_operations, self.n_operations), dtype=bool)
        s = 2+self.n_aircraft
        for idx in range(s, s+self.n_operations+1):
            self.parallel_mask[idx-s] = [int(x) for x in lines[idx].split()]

    def _init_actionspace(self):
        """
        Initialises action space as a Discrete(dim). The dimension of the Discrete action space is a function of on the
        number of machines per operation type and the number of aircraft,
        where dim = sum_{i=1}^k(n_aircraft * n_machines_k)
        """
        # Initialise Discrete actionspace
        dim = 0
        for op_type in range(0, self.n_operations+1):
            dim += self.n_aircraft * self.machines_per_op[op_type]
        self.action_space = spaces.Discrete(dim)

    def _init_observationspace(self):
        """
        Initialises the observation space as a dictionary of simple spaces.

        IDEAS:
         - number of unassigned operations
         - number of tardy flights
         - total tardiness [min]
         - utilisation rate of each machine [%]
        """

    def _init_availability(self):
        """
        Initialises the availability matrix showing which machines can be used for which aircraft operations, the matrix
        has the shape [aircraft x operations x machines].
        """
        self.availability = np.empty(shape=(self.n_aircraft, self.n_operations, self.n_machines))
        type_start = 0
        for operation_type, nmachines in enumerate(self.machines_per_op):
            next_type_start = type_start+nmachines+1

            # assign availablity and machine upper and lower bounds
            self.availability[type_start:next_type_start, :, operation_type] = 1

            # assign variables for next iteration
            type_start = next_type_start


    def _init_assignment(self):
        """
        Initialises the assignment matrix [aircraft x operations x machines] for each type of operation
        :return:
        """
        for nmachines in self.machines_per_op:
            self.assignment.append(np.empty(shape=(self.n_aircraft, self.n_operations, nmachines)))

    def _init_operation_times(self):
        """
        Performs initial time operations and generates a time conflict matrix for all operations.
        """
        self.operation_times = np.empty(shape=(self.n_aircraft, self.n_operations), dtype=dict)
        for aircraft in range(0, self.n_aircraft):
            eta = self.aircraft[aircraft]['ETA']
            for operation in range(0, self.n_operations):
                # identify preceding operations - if the current operation index < preceding index, automtically
                # calculate earliest time
                for i in range(0, self.n_operations):
                    if i == operation:
                        continue
                    else:
                        if self.parallel_mask[i, operation] and not self.parallel_mask[operation, i]:




                earliest_start = self.aircraft[aircraft]['ETA'] + Time
                self.operation_times[aircraft, operation] = {}

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

    def update_assignment(self, action):
        aircraft_index, operation_index, machine_index = self.convert_action_to_assignment(action)
        self.assignment[aircraft_index, operation_index, machine_index] = 1

    def update_availability(self, action):
        x = 5
