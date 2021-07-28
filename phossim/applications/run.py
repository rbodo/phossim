from phossim.interface.agent import AbstractAgent as Agent
from phossim.interface.environment import AbstractEnvironment as Environment
from phossim.interface.stimulation import AbstractStimulusGenerator \
    as StimulusGenerator
from phossim.interface.phosphenes import AbstractPhospheneGenerator \
    as PhospheneGenerator

# Set up pipeline.
environment = Environment()
agent = Agent()
stimulus_generator = StimulusGenerator()
phosphene_generator = PhospheneGenerator()

# Main loop.
is_done, reward, state = environment.reset()
while not is_done:

    # Transform environment state into stimulus pattern.
    state = stimulus_generator.step(state)

    # Transform stimulus pattern into phosphene pattern.
    state = phosphene_generator.step(state)

    # Take action.
    action = agent.step(state)

    # Update environment.
    is_done, reward, state = environment.step(action)

    # Train models.
    stimulus_generator.train()
    phosphene_generator.train()
    agent.train()

    # Display results.
    environment.visualize()
