using StatsBase    # for sampling
using Random, Distributions    # for continuous HMM
using LinearAlgebra

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
            error("Size of hidden state space: $J disagrees with " *
            "size of transition matrix: $(size(A, 1)) × $(size(A, 2))")
        elseif size(B) != (J, K)
            error("Size of hidden state space: $J or " *
            "size of emission space: $K disagrees with " *
            "size of emission matrix: $(size(B, 1)) × $(size(B, 2))")
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
    # Gaussian mixture model
    hidden_state_space::Array{<:Any, 1}
    initial_distribution::Array{Float64, 1}
    transition_matrix::Array{Float64, 2}
    Gaussian_center::Array{Float64, 2}
    Gaussian_deviation::Array{Float64, 2}
    mixture_coefficient::Array{Float64, 2}
    name::String
    # check parameter dimensionalities
    function Gaussian_HMM(S, μ⁰, A, M, σ², W, name = "HMM_model")
        J = length(S)    # number of hidden states
        K = size(W, 2)    # number of mixtures
        if length(μ⁰) != J
            error("Size of hidden state space: $J and " *
            "size of initial distribution: $(length(μ⁰)) mismatch")
        elseif size(A) != (J, J)
            error("Size of hidden state space: $J disagrees with " *
            "size of transition matrix: $(size(A, 1)) × $(size(A, 2))")
        elseif !all(x -> abs(x - 1.0) < 1e-12, sum(A, dims = 2))
            error("The transition matrix is not normalized")
        elseif abs(sum(μ⁰) - 1.0) ≥ 1e-12
            error("The initial distribution is not normalized")
        elseif size(M) != (J, K)
            error("The size of Gaussian center matrix: " *
            "$(size(M, 1)) × $(size(M, 2)) is incorrect " *
            "Should be $J × $K")
        elseif size(σ²) != (J, K)
            error("The size of Gaussian deviation matrix: " *
            "$(size(σ², 1)) × $(size(σ², 2)) is incorrect " *
            "Should be $J × $K")
        elseif size(W, 1) != J
            error("The size of mixture coefficient $(size(W, 1)) × $(size(W, 2)) " *
            "does not agree with the size of hidden state space $J")
        elseif !all(x -> abs(x - 1.0) < 1e-12, sum(W, dims = 2))
            error("The Gaussian mixture coefficient is not normalized")
        end
        # check Gaussian distribution parameters
        # TODO: currently not determined whether to use mixed Gaussian
        new(S, μ⁰, A, M, σ², W, name)
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
        model.Gaussian_deviation,
        model.mixture_coefficient
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
        length(model.hidden_state_space),
        size(model.mixture_coefficient, 2)
    )
end

function init_random_HMM(J, K, type = "discrete")
    # initialize an HMM with random (but valid) parameters
    # J is the number of hidden states
    # if discrete, K is the number of observed states
    # elseif continuous, K is the number of Gaussian mixtures
    # the hiddden/observed states are represented by integers

    # the transition probability
    A = rand(Float64, (J, J))
    A ./= sum(A, dims = 2) * ones(Float64, (1, J)) # normalize
    # the initial distribution
    μ⁰ = rand(Float64, J)
    μ⁰ ./= sum(μ⁰)

    if type == "discrete"
        B = rand(Float64, (J, K))
        B ./= sum(B, dims = 2) * ones(Float64, (1, K))
        model = HMM(
            collect(0:(J - 1)),    # the hidden states start from 0
            collect(1:K),
            μ⁰,
            A,
            B,
            "random_HMM"
        )
    elseif type == "Gaussian"
        M = rand(Float64, (J, K))    # Gaussian center
        # note that the deviance is not randomized!
        σ² = ones(Float64, (J, K))    # Gaussian deviation
        W = rand(Float64, (J, K))    # mixture coefficient
        W ./= sum(W, dims = 2)
        model = Gaussian_HMM(
            collect(0:(J - 1)),
            μ⁰,
            A,
            M,
            σ²,
            W,
            "random_HMM"
        )
    end
    return model
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

    μ⁰, A, M, σ², W = get_param(model)
    S, J = get_config(model)
    # Hidden state sequence
    X_idx = zeros(Int64, l)    # allocate mem
    X_idx[1] = sample(1:J, Weights(μ⁰))
    for i = 2:l
        X_idx[i] = sample(1:J, Weights(A[X_idx[i - 1], :]))
    end
    X = S[X_idx]    # convert to hidden state space
    # Observed sequence
    d = [ MixtureModel(Normal, collect(zip(M[i, :], sqrt.(σ²[i, :]))), W[i, :]) 
    for i in 1:J ]
    Y = [ rand(d[i]) for i in X_idx ]
    return X, Y
end

function b(model::HMM, j::Int64, v, ϵ = 0.01)
    # The probability that v is observed given hidden state jl
    # bⱼ(v) = P(Yₙ = v|Xₙ = sⱼ)

    B = model.emission_matrix
    V = model.emission_space
    v_idx = findfirst(V .== v)    # get the index of v
    return B[j, v_idx]
end

function b(model::Gaussian_HMM, j::Int64, v, ϵ = 0.01)
    # The continuous version

    M = model.Gaussian_center
    σ² = model.Gaussian_deviation
    W = model.mixture_coefficient
    d = MixtureModel(Normal, collect(zip(M[j, :], sqrt.(σ²[j, :]))), W[j, :])
    # take the integral around a small vicinity
    return cdf(d, v + ϵ) - cdf(d, v - ϵ)
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
    # cₙ = P(Yₙ|Y₀^{n-1}), c₀ = P(Y₀)
    # α̂ₙ(i) = P(Xₙ=sᵢ|Y₀ⁿ), α̂₀(i) = P(X₀=sᵢ|Y₀)
    C = zeros(Float64, N)
    α̂ = zeros(Float64, (J, N))
    # Initialization
    α̂[:, 1] = [ b(model, i, Y[1]) for i in 1:J ] .* μ⁰
    C[1] = sum(α̂[:, 1])
    α̂[:, 1] /= C[1]
    # Recursion
    for n = 2:N
        # notice A': summing the multiplication over columns
        α̂[:, n] = [ b(model, i, Y[n]) for i in 1:J ] .* (A' * α̂[:, n - 1])
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
    # βₙ(i) = (Y_{n+1}^N | Xₙ = sᵢ), β_N = 1
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

        # γₙ(i) is the possibility of the n_th hidden value being sᵢ
        # γₙ(i) = P(Xₙ = sᵢ|Y;θ)
        γ = C' .* α̂ .* β̂
        # ξₙ(i, j) is the possibility that the n_th hidden state is sᵢ
        # and transit to n+1_th hidden state of sⱼ
        # ξₙ(i, j) = P(Xₙ = sᵢ, X_{n + 1} = sⱼ|Y; θ)
        ξ = zeros(Float64, (J, J, N - 1))    # allocate mem
        for n  = 1:(N - 1)
            ξ[:, :, n] = α̂[:, n] * (B[:, V_idx[Y[n + 1]]] .* β̂[:, n + 1])' .* A
        end

        # update the transition matrix
        A_new = sum(ξ, dims = 3) ./ sum(γ[:, 1:(N - 1)], dims = 2)
        A_new = dropdims(A_new, dims = 3)
        # update the initial distribution
        μ⁰_new = γ[:, 1]
        # update the emission matrix
        B_new = zeros(Float64, (J, K))    # allocate mem
        for i in 1:K
            # manually add a false: the Nth col will not be accessed
            #B_new[:, i] = sum(γ[:, [Y[1:(N-1)] .== V[i]; false]], dims = 2)
            B_new[:, i] = sum(γ[:, Y .== V[i]], dims = 2)
        end
        B_new ./= sum(B_new, dims = 2)

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
    S, J, K = get_config(initial_model)
    N = length(Y)    # size of observed sequence
    # the parameters are already initialized inside initial_model
    model = initial_model   # copy the initial model
    Δ = Inf    # difference between the current and previous log-likelihood
    lnL = log_likelihood(model, Y)    # initial log likelihood
    while(Δ > threshold)
        # variable shorthands
        μ⁰, A, M, σ², W = get_param(model)
        # the forward and backward matrix
        β̂, α̂ , C = backward(model, Y)

        # γₙ(i) is the possibility of the n_th hidden value being sᵢ
        # ξₙ(i, j) is the possibility that the n_th hidden state is sᵢ
        # and transit to n+1_th hidden state of sⱼ
        γ = zeros(Float64, (J, K, N))    # allocate mem
        for n = 1:N
            p = C[n] * α̂[:, n] .* β̂[:, n]
            w = zeros(Float64, (J, K))
            for j = 1:J
                for k = 1:K
                    w[j, k] = W[j, k] * pdf(Normal(M[j, k], sqrt(σ²[j, k])), Y[n])
                end
            end
            w ./= sum(w, dims = 2)
            γ[:, :, n] = p .* w
        end
        ξ = zeros(Float64, (J, J, N - 1))    # allocate mem
        for n  = 1:(N - 1)
            ξ[:, :, n] = α̂[:, n] * ([ b(model, j, Y[n + 1]) for j in 1:J ] .* β̂[:, n + 1])' .* A
        end

        # update the transition matrix
        A_new = sum(ξ, dims = 3) ./ sum(γ[:, :, 1:(N - 1)], dims = (2, 3))
        A_new = dropdims(A_new, dims = 3)    # remove singleton dimension

        # update the initial distribution
        μ⁰_new = sum(γ[:, :, 1], dims = 2)
        μ⁰_new = vec(μ⁰_new)

        # update the mixture coefficient
        W_new = sum(γ, dims = 3)
        W_new = dropdims(W_new, dims = 3)
        W_new ./= sum(W_new, dims = 2)

        # update the mean of Gaussian distribution
        M_new = zeros(Float64, (J, K))
        for k in 1:K
            M_new[:, k] = (γ[:, k, :] * Y) ./ sum(γ[:, k, :], dims = 2)
        end

        # update the deviation of Gaussian
        σ²_new = zeros(Float64, (J, K))
        for i = 1:J
            for k in 1:K
                # σ²_new[i, k] = sum(((Y .- M[i, k]).^2 .* γ[i, k, :])[1:(N - 1)]) / 
                # sum(γ[i, k, 1:(N - 1)])
                σ²_new[i, k] = sum((Y .- M[i, k]).^2 .* γ[i, k, :]) / sum(γ[i, k, :])
            end
        end

        # update the HMM model
        model = Gaussian_HMM(S, μ⁰_new, A_new, M_new, σ²_new, W_new, initial_model.name)

        # update Δln(L)
        new_lnL = log_likelihood(model, Y)
        Δ = abs(lnL - new_lnL)
        lnL = new_lnL
    end
    return model
end

function viterbi(model::AbstractHMM, Y::AbstractArray)
    # The Viterbi algorithm to infer the hidden states

    # variable shorthands
    μ⁰ = model.initial_distribution
    A = model.transition_matrix
    S = model.hidden_state_space
    J = length(S)    # size of hidden state space
    N = length(Y)    # length of observed seq
    # Φₙ(i) is the maximum log-likelihood of the hidden states
    # from 0 to n-1, ending with Xₙ = sᵢ
    # Φₙ(i) = max{i₀ to i_{n-1}} lnP(Y₀^n, X₀^{n-1}, Xₙ = sᵢ)
    Φ = zeros(Float64, (J, N))
    Ψ = zeros(Float64, (J, N))    # for traceback
    # Initialization
    Φ[:, 1] = log.(μ⁰) + log.([ b(model, i, Y[1]) for i in 1:J ])
    # Recursion
    for n = 2:N
        for i = 1:J
            idx = argmax(log.(A[:, i]) + Φ[:, n - 1])
            Ψ[i, n] = idx
            Φ[i, n] = log(b(model, i, Y[n])) + log(A[idx, i]) + Φ[idx, n - 1]
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
