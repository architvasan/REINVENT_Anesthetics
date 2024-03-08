import csv
import pytest

from reinvent.runmodes.samplers.run_sampling import run_sampling
from reinvent.runmodes.utils.helpers import set_torch_device


# FIXME: the problem here is that we need to read this from JSON but we can use
#        json_config as a parameter to a function


def check_csv_columns_in_file(filename, num_columns):
    with open(filename, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")

        for i, row in enumerate(reader):
            if i == 0:
                continue

            if len(row) != num_columns:
                return False

            float(row[-1])

    return i


@pytest.fixture
def param(request, json_config):
    params = {
        "reinvent": {
            "prior": json_config["PRIOR_PATH"],
            "num_smiles": 20,
            "smiles_multiplier": 1,
            "smiles_file": None,
            "sample_strategy": None,
            "num_cols": 2
        },
        "libinvent": {
            "prior": json_config["LIBINVENT_PRIOR_PATH"],
            "num_smiles": 10,
            "smiles_multiplier": 2,
            "smiles_file": json_config["LIBINVENT_SMILES_SCAFFOLDS"],
            "sample_strategy": None,
            "num_cols": 4
        },
        "linkinvent": {
            "prior": json_config["LINKINVENT_PRIOR_PATH"],
            "num_smiles": 10,
            "smiles_multiplier": 1,
            "smiles_file": json_config["LINKINVENT_SMILES_WARHEADS"],
            "sample_strategy": None,
            "num_cols": 4
        },
        "mol2mol-multi": {
            "prior": json_config["MOLFORMER_PRIOR_PATH"],
            "num_smiles": 5,
            "smiles_multiplier": 3,
            "smiles_file": json_config["MOLFORMER_SMILES_SET_PATH"],
            "sample_strategy": "multinomial",
            "num_cols": 4
        },
        "mol2mol-beam": {
            "prior": json_config["MOLFORMER_PRIOR_PATH"],
            "num_smiles": 1,
            "smiles_multiplier": 3,
            "smiles_file": json_config["MOLFORMER_SMILES_SET_PATH"],
            "sample_strategy": "beamsearch",
            "num_cols":4
        },
    }

    return params[request.param]


IDs = ["reinvent", "libinvent", "linkinvent", "mol2mol-multi", "mol2mol-beam"]


@pytest.fixture
def setup(tmp_path, pytestconfig):
    device = pytestconfig.getoption("device")
    set_torch_device(device)

    output_filename = tmp_path / "samplers.smi"

    config = {
        "parameters": {
            "output_file": output_filename,
            "batch_size": 128,
            "unique_molecules": False,
            "randomize_smiles": True,
        }
    }

    return output_filename, config


@pytest.mark.integration
@pytest.mark.parametrize("param", IDs, ids=IDs, indirect=True)
def test_run_sampling_with_likelihood(param, setup, pytestconfig):
    output_filename, config = setup
    device = pytestconfig.getoption("device")

    cfg_params = config["parameters"]
    cfg_params["model_file"] = param["prior"]
    cfg_params["num_smiles"] = param["num_smiles"]
    cfg_params["smiles_file"] = param["smiles_file"]
    cfg_params["sample_strategy"] = param["sample_strategy"]
    num_cols = param["num_cols"]

    run_sampling(config, device)

    num_lines = check_csv_columns_in_file(output_filename, num_columns=num_cols)
    num_smiles = param["smiles_multiplier"] * param["num_smiles"]

    assert num_lines == num_smiles
