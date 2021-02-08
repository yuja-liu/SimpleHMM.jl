# Simple Hidden Markov Model (HMM)

## Quick start

### Install

Start the Julia REPL. Press `]` to enter the Pkg mode.
Then execute the following line:

```
Pkg> add SimpleHMM
```

By now `SimpleHMM` is all installed and ready to use!

In your Julia code, load the package with:

```julia
using SimpleHMM
```

### Read an HMM model (with parameters) from a JSON file

```julia
# Load the example model
model_path = joinpath(dirname(pathof(SimpleHMM)), "../data", "example_model.json")
model = HMM_from_json(model_path)
```

### Generate an emitted sequence from the model

```julia
# Sequence length = 100
_, emitted_seq = emit(model, 100)
# Print the sequence:
println(emitted_seq)
```

```
[3, 3, 3, 3, 3, 3, 4, 5, 3, 4, 2, 2, 1, 3, 1, 3, 2, 4, 4, 5, 3, 4, 2, 3, 3, 3, 4, 2, 1, 2, 2, 2, 1, 2, 3, 3, 3, 4, 2, 4, 3, 3, 3, 5, 2, 3, 3, 4, 2, 4, 4, 3, 4, 5, 4, 3, 3, 4, 4, 5, 4, 3, 3, 4, 4, 4, 3, 5, 2, 2, 2, 1, 4, 2, 4, 3, 4, 4, 3, 4, 3, 2, 4, 4, 4, 2, 3, 2, 5, 2, 4, 2, 3, 3, 1, 2, 4, 5, 3, 4]
```

We can also calculate the log-likelihood of this particular sequence being observed:

```julia
log_likelihood(model, emitted_seq)
```

```
-139.14822148811922
```

### Infer parameters from an observed sequence

First, we will initialize an HMM as the start point of inference:

```julia
# Initialize a HMM with random parameters
# The size of the hidden state space is 2
# The size of the observed state space is 5
initial_model = init_random_HMM(2, 5)
# Check the emission probability matrix:
display(initial_model.emission_matrix)
```

```
2×5 Array{Float64,2}:
 0.273413  0.197861  0.117856   0.14479   0.26608
 0.127547  0.211586  0.0694058  0.366105  0.225356
```

Then, use the previously generated observed sequence to train HMM

```julia
# The new parameters are stored in the "new model"
new_model = baum_welch(initial_model, emitted_seq)
# Examine the trained emission probabilities:
display(new_model.emission_matrix)
```

```
2×5 Array{Float64,2}:
 0.285027     0.545211  0.146414  0.0224742  0.000874392
 3.35735e-11  0.133287  0.391617  0.373998   0.101098
 ```

### Infer the hidden states from an observed sequence

```julia
# Let's use the trained model to infer the hidden states
hidden_seq = viterbi(new_model, emitted_seq)
# check it out
println(hidden_seq)
```

```
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
```

### HMM with continuous emission probabilities

SimpleHMM also supports models with continuous emission probabilities.
The emission probability conforms a Gaussian mixture model and the number of mixtures can be adjusted.

```julia
# Initialized a random HMM with continuous emission
# The size of hidden states space is 2
# The number of mixtures for the Gaussian mixture model is 2
continuous_model = init_random_HMM(2, 2, "Gaussian")

# The emit an observed sequence or infer the parameters/hidden states
# just like we did to the discrete model
```

## Reference

This Julia package is the product of the Genomic Sequence Analysis
module, by Dr Aylwyn Scally, as part of the Cambridge MPhil
in Computational Biology programme.

```
[1]L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition,” Proceedings of the IEEE, vol. 77, no. 2, pp. 257–286, Feb. 1989, doi: 10.1109/5.18626.
[2]A. N. of Loc Nguyen, “Continuous Observation Hidden Markov Model,” Revista Kasmera, vol. 44, pp. 65–149, Jun. 2016.
```
