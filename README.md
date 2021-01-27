# Simple HMM module for Julia

## Quick start

### Install

Start the Julia REPL. Press `]` to enter the Pkg mode.
Then execute the following line:

```
Pkg> add https://github.com/RandolphLiu/HMMlib.jl.git
```

By now `HMMlib` is all installed and ready to use!

In your Julia code, load the package with:

```julia
using HMMlib
```

### Read an HMM model (with parameters) from a JSON file

```julia
# load the example model
model_path = joinpath(dirname(pathof(HMMlib)), "../data", "example_model.json")
model = HMM_from_json(model_path)
```

### Generate an emitted sequence from the model

```julia
# sequence length = 100
_, emitted_seq = emit(model, 100)
```

### Infer parameters from an observed sequence

```julia
# the new parameters are stored in the "new model"
new_model = baum_welch(initial_model, emitted_seq)
```

### Infer the hidden states from an observed sequence

```julia
hidden_seq = viterbi(model, emitted_seq)
```
