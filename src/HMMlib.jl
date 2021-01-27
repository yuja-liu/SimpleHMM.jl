module HMMlib
# Simple HMM library

using StatsBase    # for sampling
using Random, Distributions    # for continuous HMM
using LinearAlgebra

export 
HMM, Gaussian_HMM, emit, baum_welch, viterbi, log_likelihood,
get_param, get_config,
HMM_from_json

# include the io utilities
include("io.jl")

# Definition of HMM model type
struct HMM
    # the states can be string, int, etc.
    hidden_state_space::Array{<:Any, 1}
    emission_space::Array{<:Any, 1}
    initial_distribution::Array{Float64, 1}
    transition_matrix::Array{Float64, 2}
    emission_matrix::Array{Float64, 2}
    name::String
    # check parameter dimensionalities
    function HMM(S, V, μ⁰, A, B, name = "HMM_model")
        J = length(S)    # number of hidden states
        K = length(V)    # number of emitted states
        if length(μ⁰) != J
            error("Size of hidden state space: $J and 
            size of initial distribution: $(length(μ⁰)) mismatch")
        elseif size(A) != (J, J)
            error("Size of hidden state space: $J disagrees with 
            size of transition matrix: $(size(A, 1)) × $(size(A, 2))")
        elseif size(B) != (J, K)
            error("Size of hidden state space: $J or
            size of emission space: $K disagrees with
            size of emission matrix: $(size(B, 1)) × $(size(B, 2))")
        elseif !all(x -> abs(x - 1.0) < 1e-12, sum(A, dims = 2))
            error("The transition matrix is not normalized")
        elseif !all(x -> abs(x - 1.0) < 1e-12, sum(B, dims = 2))
            error("The emission matrix is not normalized")
        elseif abs(sum(μ⁰) - 1.0) ≥ 1e-12
            error("The initial distribution is not normalized")
        end
        new(S, V, μ⁰, A, B, name)
    end
end

# Definition of HMM with continuous emission distribution
struct Gaussian_HMM
    hidden_state_space::Array{<:Any, 1}
    initial_distribution::Array{Float64, 1}
    transition_matrix::Array{Float64, 2}
    Gaussian_center::Array{Float64, 1}
    Gaussian_deviation::Array{Float64, 1}
    name::String
    # check parameter dimensionalities
    function Gaussian_HMM(S, μ⁰, A, m, σ², name = "HMM_model")
        J = length(S)    # number of hidden states
        if length(μ⁰) != J
            error("Size of hidden state space: $J and 
            size of initial distribution: $(length(μ⁰)) mismatch")
        elseif size(A) != (J, J)
            error("Size of hidden state space: $J disagrees with 
            size of transition matrix: $(size(A, 1)) × $(size(A, 2))")
        elseif !all(x -> abs(x - 1.0) < 1e-12, sum(A, dims = 2))
            error("The transition matrix is not normalized")
        elseif abs(sum(μ⁰) - 1.0) ≥ 1e-12
            error("The initial distribution is not normalized")
        end
        # check Gaussian distribution parameters
        # TODO: currently not determined whether to use mixed Gaussian
        new(S, μ⁰, A, m, σ², name)
    end
end

# All HMM
AbstractHMM = Union{HMM, Gaussian_HMM}

function get_param(model::HMM)
    # A helper function to return the parameters

    return (
        model.initial_distribution,
        model.transition_matrix,
        model.emission_matrix
    )
end

function get_param(model::Gaussian_HMM)
    # Get the parameters of Gaussian HMM

    return (
        model.initial_distribution,
        model.transition_matrix,
        model.Gaussian_center,
        model.Gaussian_deviation
    )
end

function get_config(model::HMM)
    # A helper function to get model configurations

    return (
        model.hidden_state_space,
        model.emission_space,
        length(model.hidden_state_space),
        length(model.emission_space)
    )
end

function get_config(model::Gaussian_HMM)
    # Get configurations for Gaussian HMM

    return (
        model.hidden_state_space,
        length(model.hidden_state_space)
        # TODO: if using mixture distribution, also return number of categories
    )
end

function emit(model::HMM, l::Int64)
    # generate an emitted sequence and hidden state sequence
    # with given HMM model

    # parameter shorthands
    μ⁰, A, B = get_param(model)
    S, V, J, _ = get_config(model)
    # generate a sequence of emitted states and hidden states with given HMM model
    X = fill(S[1], l)    # allocate mem for hidden states
    # convert state to index
    S_idx = Dict( S[i] => i for i in 1:J )
    # generate X₀
    X[1] = sample(S, Weights(μ⁰))
    # generate the remaining hidden states
    for i = 2:l
        X[i] = sample(S, Weights(A[S_idx[X[i - 1]], :]))
    end
    # generate emitted states
    Y = [ sample(V, Weights(B[S_idx[X[i]], :])) for i in 1:l ]
    return X, Y
end

function emit(model::Gaussian_HMM, l::Int64)
    # Generate observed sequence - continuous version

    μ⁰, A, m, σ² = get_param(model)
    S, J = get_config(model)
    # Hidden state sequence
    X_idx = zeros(Int64, l)    # allocate mem
    X_idx[1] = sample(1:J, Weights(μ⁰))
    for i = 2:l
        X_idx[i] = sample(1:J, Weights(A[X_idx[i - 1], :]))
    end
    X = S[X_idx]    # convert to hidden state space
    # Observed sequence
    d = [ Normal(m[i], sqrt(σ²[i])) for i in 1:J ]
    Y = [ rand(d[i]) for i in X_idx ]
    return X, Y
end

function b(model::AbstractHMM, j::Int64, v, ϵ = 0.01)
    # The probability that v is observed given hidden state jl
    # bⱼ(v) = P(Yₙ = v|Xₙ = sⱼ)

    if typeof(model) == HMM
        # Discrete HMM
        B = model.emission_matrix
        V = model.emission_space
        v_idx = findfirst(V .== v)    # get the index of v
        return B[j, v_idx]
    elseif typeof(model) ==  Gaussian_HMM
        # Continuous observation HMM
        m = model.Gaussian_center
        σ² = model.Gaussian_deviation
        d = Normal(m[j], sqrt(σ²[j]))
        # take the integral around a small vicinity
        return cdf(d, v + ϵ) - cdf(d, v - ϵ)
    end
end

function forward(model::AbstractHMM, Y::AbstractArray)
    # the forward algorithm

    # variable shorthands
    μ⁰ = model.initial_distribution
    A = model.transition_matrix
    S = model.hidden_state_space
    J = length(S)    # size of hidden state space
    N = length(Y)    # length of observed seq
    # Allocate mem
    # Cₙ = P(Yₙ|Y₀ⁿ⁻¹), C₀ = P(Y₀)
    # α̂ₙ(i) = P(Xₙ=sᵢ|Y₀ⁿ), α̂₀(i) = P(X₀=sᵢ|Y₀)
    C = zeros(Float64, N)
    α̂ = zeros(Float64, (J, N))
    # Initialization
    α̂[:, 1] = [ b(model, i, Y[1]) for i in 1:J ] .* μ⁰
    C[1] = sum(α̂[:, 1])
    α̂[:, 1] /= C[1]
    # Recursion
    for n = 2:N
        # notice Aᵀ: summing the multiplication over columns
        α̂[:, n] = [ b(model, i, Y[n]) for i in 1:J ] .* (transpose(A) * α̂[:, n - 1])
        C[n] = sum(α̂[:, n])
        α̂[:, n] /= C[n]
    end
    return(α̂, C)
end

function log_likelihood(model::AbstractHMM, Y::AbstractArray)
    # Calculate the log-likelihood of observed (emitted) sequence
    # with given HMM model

    _, C = forward(model, Y)
    sum([ log(c) for c in C ])
end

function backward(model::AbstractHMM, Y::AbstractArray)
    # The backward algorithm

    # variable shorthands
    μ⁰ = model.initial_distribution
    A = model.transition_matrix
    S = model.hidden_state_space
    J = length(S)    # size of hidden state space
    N = length(Y)    # length of observed seq
    # Allocate mem
    # βₙ(i) = (Yᴺₙ₊₁ | Xₙ = sᵢ), β_N = 1
    β̂ = zeros(Float64, (J, N))
    # get the C terms
    α̂, C = forward(model, Y)
    # Initialization
    β̂[:, N] .= 1/C[N]
    # Recursion
    for n = (N - 1):-1:1
        β̂[:, n] = A * ([ b(model, i, Y[n + 1]) for i in 1:J ] .* β̂[:, n + 1]) ./ C[n]
    end
    # also returns α̂ since it's free
    return β̂, α̂, C
end

function baum_welch(initial_model::HMM, Y::AbstractArray, threshold = 1e-2)
    # The Baum-Welch algorithm to infer the parameters of HMM

    # shorthands of variables
    S, V, J, K = get_config(initial_model)
    N = length(Y)    # size of observed sequence
    # convert state to index
    V_idx = Dict( V[i] => i for i in 1:K )
    # the parameters are already initialized inside initial_model
    model = initial_model   # copy the initial model
    Δ = Inf    # difference between the current and previous log-likelihood
    lnL = log_likelihood(model, Y)    # initial log likelihood
    while(Δ > threshold)
        # variable shorthands
        μ⁰, A, B = get_param(model)
        # the forward and backward matrix
        β̂, α̂ , C = backward(model, Y)

        # update the transition matrix
        # the expcted count of i -> j transition
        En = zeros(Float64, (J, J))
        for i in 1:J
            for j in 1:J
                En[i, j] = A[i, j] * α̂[i, 1:(N - 1)] ⋅ 
                (B[j, [ V_idx[Y[k]] for k in 2:N ]] .* β̂[j, 2:N])
            end
        end
        A_new = En ./ (sum(En, dims = 2) * ones(Float64, (1, J)))    # normalize

        # update the initial distribution
        μ⁰_new = α̂[:, 1] .* (A * (B[:, V_idx[Y[2]]] .* β̂[:, 2]))

        # update the emission matrix
        # define γₙ(i) = P(Xₙ = sᵢ|Y;θ)
        γ = (ones(Float64, J) * reshape(C, (1, N))) .* α̂ .* β̂
        B_new = zeros(Float64, (J, K))    # allocate mem
        for i in 1:K
            # manually add a false: the Nth col will not be accessed
            B_new[:, i] = sum(γ[:, [Y[1:(N-1)] .== V[i]; false]], dims = 2)
        end
        B_new ./= sum(B_new, dims = 2) * ones(Float64, (1, K))

        # update the HMM model
        model = HMM(S, V, μ⁰_new, A_new, B_new, initial_model.name)

        # update Δln(L)
        new_lnL = log_likelihood(model, Y)
        Δ = abs(lnL - new_lnL)
        lnL = new_lnL
    end
    return model
end

function baum_welch(initial_model::Gaussian_HMM, Y::AbstractArray, threshold = 1e-2)
    # Continuous version of Baum-Welch

    # shorthands of variables
    S, J = get_config(initial_model)
    N = length(Y)    # size of observed sequence
    # the parameters are already initialized inside initial_model
    model = initial_model   # copy the initial model
    Δ = Inf    # difference between the current and previous log-likelihood
    lnL = log_likelihood(model, Y)    # initial log likelihood
    while(Δ > threshold)
        # variable shorthands
        μ⁰, A, m, σ² = get_param(model)
        # the forward and backward matrix
        β̂, α̂ , C = backward(model, Y)

        # update the transition matrix
        # the expcted count of i -> j transition
        En = zeros(Float64, (J, J))
        for i in 1:J
            for j in 1:J
                En[i, j] = A[i, j] * α̂[i, 1:(N - 1)] ⋅ 
                ([ b(model, j, Y[n]) for n in 2:N ] .* β̂[j, 2:N])
            end
        end
        A_new = En ./ (sum(En, dims = 2) * ones(Float64, (1, J)))    # normalize
        
        # define γₙ(i) = P(Xₙ = sᵢ|Y;θ)
        # the probability that n^th state being sᵢ, given observed seq
        γ = (ones(Float64, J) * reshape(C, (1, N))) .* α̂ .* β̂

        # update the initial distribution
        μ⁰_new = γ[:, 1]

        # update the mean of Gaussian distribution
        m_new = (γ * Y) ./ sum(γ, dims = 2)
        m_new = vec(m_new)    # reshape to a column vector

        # update the deviation of Gaussian
        σ²_new = zeros(Float64, J)
        for i = 1:J
            σ²_new[i] = sum(((Y .- m[i]).^2 .* γ[i, :])[1:(N - 1)]) / sum(γ[i, 1:(N - 1)])
        end

        # update the HMM model
        model = Gaussian_HMM(S, μ⁰_new, A_new, m_new, σ²_new, initial_model.name)

        # update Δln(L)
        new_lnL = log_likelihood(model, Y)
        Δ = abs(lnL - new_lnL)
        lnL = new_lnL
    end
    return model
end

function viterbi(model::HMM, Y::AbstractArray)
    # The Viterbi algorithm to infer the hidden states

    # variable shorthands
    μ⁰, A, B = get_param(model)
    S, V, J, K = get_config(model)
    N = length(Y)    # length of observed sequence
    # convert state to index
    V_idx = Dict( V[i] => i for i in 1:K )
    # Φₙ(i) is the maximum log-likelihood of the hidden states
    # from 0 to n-1, ending with Xₙ = sᵢ
    # Φₙ(i) = max{i₀ to iₙ₋₁} lnP(Y₀ⁿ, X₀ⁿ⁻¹, Xₙ = sᵢ)
    Φ = zeros(Float64, (J, N))
    Ψ = zeros(Float64, (J, N))    # for traceback
    # Initialization
    Φ[:, 1] = log.(μ⁰) + log.(B[:, V_idx[Y[1]]])
    # Recursion
    for n = 2:N
        for i = 1:J
            idx = argmax(log.(A[:, i]) + Φ[:, n - 1])
            Ψ[i, n] = idx
            Φ[i, n] = log(B[i, V_idx[Y[n]]]) + log(A[idx, i]) + Φ[idx, n - 1]
        end
    end
    # traceback
    i_max = zeros(Int64, N)
    i_max[N] = argmax(Φ[:, N])
    for n = (N - 1):-1:1
        i_max[n] = Ψ[i_max[n + 1], n + 1]
    end
    # get Sᵢ
    X_max = S[i_max]
    return(X_max)
end

end    # module