module SimpleHMM

# exposed interface
export 
HMM, Gaussian_HMM, emit, baum_welch, viterbi, log_likelihood,
get_param, get_config, init_random_HMM,
HMM_from_json, HMM_to_json

# include the io utilities
# n.b. Base.jl is alreaded included by IO.jl
include("IO.jl")

end    # end of module