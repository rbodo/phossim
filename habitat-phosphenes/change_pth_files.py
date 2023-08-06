import torch

def delete_fields_from_state_dict(input_path, output_path, fields_to_delete):
    # Load the original state dictionary from the .pth file
    state_dict = torch.load(input_path, map_location=torch.device('cpu'))

    # Delete the specified fields from the state dictionary
    for field in fields_to_delete:
        del state_dict['state_dict'][field]

    # Save the modified state dictionary to a new .pth file
    torch.save(state_dict, output_path)

if __name__ == "__main__":
    # Specify the input .pth file path and output file path
    input_pth_file = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/gibson-rgbd-best.pth"
    output_pth_file = "/home/carsan/Internship/PyCharm_projects/habitat-lab/data/gibson-rgbd-best2.pth"

    # List of fields to delete from the state dictionary
    fields_to_delete = ["actor_critic.net.tgt_embeding.weight", "actor_critic.net.tgt_embeding.bias"]

    # Call the function to delete the specified fields and save the modified state dictionary
    delete_fields_from_state_dict(input_pth_file, output_pth_file, fields_to_delete)
