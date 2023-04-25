import matplotlib
import numpy as np
import plotly.express as px

def machine_assignment_dict(assignment, machines_per_op, aircraft_info, operation_times):
    """
    Converts a scheduling assignment to a dictionary suited for usage with ff.create_gantt()
    """
    assignment_dict = []
    machine_index = 0
    for op_type in range(len(machines_per_op)):
        for nmachine in range(machines_per_op[op_type]):
            machine_name = f'Machine {op_type}.{nmachine}'

            # gather vector of aircraft assignments for this machine and operation type
            assignments = np.where(assignment[:, op_type, machine_index])[0]
            for aircraft_idx in assignments:
                assignment_dict.append({'Machine': machine_name,
                                        'Start': operation_times[aircraft_idx, op_type]['Scheduled Start'],
                                        'Finish': operation_times[aircraft_idx, op_type]['Scheduled End'],
                                        'Aircraft': f'{aircraft_info[aircraft_idx]["REG"]}',
                                        'Delay': 0})


            machine_index += 1
    return assignment_dict

def render_schedule(assignment, machines_per_op, aircraft_info, operation_times):
    assignment_dict = machine_assignment_dict(assignment, machines_per_op, aircraft_info, operation_times)
    fig = px.timeline(assignment_dict, x_start='Start', x_end='Finish', y='Machine', color='Aircraft')
    fig.show()