# IO utilities for HMM.jl

import JSON
include("HMM.jl")

# load model from json
function HMM_from_json(json_path)
    json_dict = JSON.parsefile(json_path)
    J = length(json_dict["hidden_state_space"])
    A = reshape(json_dict["transition_matrix"], (J, :))
    B = reshape(json_dict["emission_matrix"], (J, :))
    HMMlib.HMM(
        json_dict["hidden_state_space"],
        json_dict["emission_space"],
        json_dict["initial_distribution"],
        A,
        B,
        json_dict["name"]
    )
end