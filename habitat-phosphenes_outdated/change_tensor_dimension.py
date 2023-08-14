import torch

def modify_checkpoint_checkpoint(checkpoint_path, output_path, target_rnn_shape):
    # Load the original state dictionary from the checkpoint file
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Modify the RNN layer in the state dictionary to match the target shape
    state_dict['state_dict']['actor_critic.net.state_encoder.rnn.weight_ih_l0'] = state_dict['state_dict']['actor_critic.net.state_encoder.rnn.weight_ih_l0'][:, :target_rnn_shape[1]]

    # Save the modified state dictionary to a new checkpoint file
    torch.save(state_dict, output_path)

if __name__ == "__main__":
    # Specify the original checkpoint file path and output modified checkpoint file path
    checkpoint_path = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/pretrained_models/gibson-rgbd-best2.pth"
    output_path = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/pretrained_models/gibson-rgbd-best2.pth"

    # Specify the target shape for the RNN layer in the current model (torch.Size([1536, 544]))
    target_rnn_shape = (1536, 544)

    # Call the function to modify the checkpoint and save the modified version
    modify_checkpoint_checkpoint(checkpoint_path, output_path, target_rnn_shape)