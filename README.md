# Simple HMM module for Julia

## Quick start

### Load the module

```julia
include("HMM.jl")
```

### Read a HMM model (with parameters) from a JSON file

```julia
include("io.jl")
model = HMM_from_json("example_model.json")
```

### Generate an emitted sequence from the model

```julia
# sequence length = 100
_, emitted_seq = HMMlib.emit(model, 100)
```

### Infer parameters from an observed sequence

```julia
# the new parameters are stored in the "new model"
new_model = HMMlib.baum_welch(initial_model, emitted_seq)
```

### Infer the hidden states from an observed sequence

```julia
X = HMMlib.viterbi(model, emitted_seq)
```
