from phossim.implementation.environment.neurosmash.environment import Environment
from phossim.implementation.environment.neurosmash.agent import Agent, get_model, train

if __name__ == '__main__':

    # Environment #
    ###############

    ip = '127.0.0.1'  # Ip address that the TCP/IP interface listens to.
    port = 13000  # Port number that the TCP/IP interface listens to.
    resolution = 96
    timescale = 5
    num_actions = 4
    input_shape = (resolution, resolution, 3)

    environment = Environment(ip, port, resolution, timescale)
    environment.setup()

    # Model #
    #########

    model = get_model(input_shape, num_actions)

    # Training #
    ############

    train(environment, model)

    # Testing #
    ###########

    agent = Agent(model)
    state = environment.reset()

    while not environment.is_done:
        action = agent.step(state)
        environment.step(action)

    print()
