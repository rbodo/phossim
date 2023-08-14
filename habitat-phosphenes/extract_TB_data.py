from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

if __name__ == "__main__":
    input_file = "/scratch/big/home/carsan/Data/phosphenes/habitat/tb/eval/eval_NPT_GPS_Original_black/events.out.tfevents.1690838930.mars.2235472.0"
    data = parse_tensorboard(input_file, ["eval_metrics/distance_to_goal","eval_metrics/spl","eval_metrics/success"])

    print("END")