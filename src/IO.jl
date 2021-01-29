# IO utilities for HMM.jl

import JSON
include("Base.jl")    # definitions of HMM

# load model from json
function HMM_from_json(json_path::String)
    json_dict = JSON.parsefile(json_path)
    HMM_type = json_dict["type"]
    J = length(json_dict["hidden_state_space"])

    if HMM_type == "discrete"
        A = reshape(json_dict["transition_matrix"], (J, :))
        B = reshape(json_dict["emission_matrix"], (J, :))
        model = HMM(
            json_dict["hidden_state_space"],
            json_dict["emission_space"],
            json_dict["initial_distribution"],
            A,
            B,
            json_dict["name"]
        )
    elseif HMM_type == "Gaussian"
        A = reshape(json_dict["transition_matrix"], (J, :))
        M = reshape(json_dict["Gaussian_center"], (J, :))
        σ² = reshape(json_dict["Gaussian_deviation"], (J, :))
        W = reshape(json_dict["mixture_coefficient"], (J, :))
        model = Gaussian_HMM(
            json_dict["hidden_state_space"],
            json_dict["initial_distribution"],
            A,
            M,
            σ²,
            W,
            json_dict["name"]
        )
    else
        error("HMM type not found. Type can be either " *
        "\"discrete\" or \"Gaussian\".")
    end
    return model
end

# save model to json
function HMM_to_json(model::HMM, json_path::String)
    HMM_dict = Dict(
        "type" => "discrete",
        "hidden_state_space" => model.hidden_state_space,
        "emission_space" => model.emission_space,
        "initial_distribution" => model.initial_distribution,
        "transition_matrix" => vec(model.transition_matrix),
        "emission_matrix" => vec(model.emission_matrix),
        "name" => model.name
    )
    open(json_path, "w") do f
        JSON.print(f, HMM_dict)
    end
end

function HMM_to_json(model::Gaussian_HMM, json_path::String)
    HMM_dict = Dict(
        "type" => "Gaussian",
        "hidden_state_space" => model.hidden_state_space,
        "initial_distribution" => model.initial_distribution,
        "transition_matrix" => vec(model.transition_matrix),
        "Gaussian_center" => vec(model.Gaussian_center),
        "Gaussian_deviation" => vec(model.Gaussian_deviation),
        "mixture_coefficient" => vec(model.mixture_coefficient),
        "name" => model.name
    )
    open(json_path, "w") do f
        JSON.print(f, HMM_dict)
    end
end