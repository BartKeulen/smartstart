from smartstart.agents.counter import Counter
from smartstart.agents.mbrl import MBRL
from smartstart.agents.qlearning import QLearning
from smartstart.agents.rmax import RMax
from smartstart.agents.smartstart import SmartStart
from smartstart.agents.valueiteration import ValueIteration, TransitionModel, RewardFunction

agents = {
    'QLearning': QLearning,
    'RMax': RMax,
    'MBRL': MBRL,
    'SmartStart': SmartStart
}