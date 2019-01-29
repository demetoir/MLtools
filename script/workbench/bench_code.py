# -*- coding:utf-8 -*-


from script.util.deco import deco_timeit
# from slackbot.SlackBot import deco_slackbot

# bprint = print
# logger = Logger('bench_code', level='INFO', )
# print = logger.info
# pprint = pprint_logger(print)
from script.workbench.dqn import dqn_cartpole

from script.workbench.q_network import qnetwork_frozenlake, qnetwork_cartpole
from script.workbench.q_learning import q_learning_frozenlake

@deco_timeit
# @deco_slackbot('./slackbot/tokens/ml_bot_token', 'mltool_bot')
def main():
    # for i in range(11233123):
    #     print(i)
    # qnetwork_cartpole()
    dqn_cartpole()
    # q_learning_frozenlake()




    pass
