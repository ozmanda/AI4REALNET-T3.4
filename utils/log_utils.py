from typing import NamedTuple, Dict
from collections import namedtuple

LogField = namedtuple('LogField', ('data', 'plot', 'x_axis', 'divide_by'))

def init_logger() -> Dict[str, NamedTuple]:
    """
    Initialises a dictionary of LogField NamedTuple objects
    """
    log = dict()
    log['epoch'] = LogField(list(), False, None, None)
    log['reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['enemy_reward'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['success'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['steps_taken'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['add_rate'] = LogField(list(), True, 'epoch', 'num_episodes')
    log['comm_action'] = LogField(list(), True, 'epoch', 'num_steps')
    log['enemy_comm'] = LogField(list(), True, 'epoch', 'num_steps')
    log['value_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['action_loss'] = LogField(list(), True, 'epoch', 'num_steps')
    log['entropy'] = LogField(list(), True, 'epoch', 'num_steps')
    return log
