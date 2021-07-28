from phossim.applications.atari.breakout.environment import Environment
from phossim.applications.atari.breakout.agent import Agent, get_model, train

if __name__ == '__main__':

    # Environment #
    ###############

    num_actions = 4
    input_shape = (84, 84, num_actions)

    environment = Environment()
    environment.setup()

    # Model #
    #########

    model_q = get_model(input_shape, num_actions)
    model_target = get_model(input_shape, num_actions)

    # Training #
    ############

    train(environment, model_q, model_target)

    # Testing #
    ###########

    agent = Agent(model_q)
    state = environment.reset()

    while not environment.is_done:
        action = agent.step(state)
        environment.step(action)

    print()
