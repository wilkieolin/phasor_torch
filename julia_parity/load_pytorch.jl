"""
load_pytorch.jl — load PyTorch-generated HDF5 weights into Lux NamedTuples.

The companion writer is phasor_torch/weights.py. Schema:
  /<layer_path>/<param_name>          Float32 dataset

Top-level helpers in this module:
  load_params(path, layer_path)  -> NamedTuple of params for one layer group
  load_io_pair(path)             -> (inputs::Dict{String,Array}, outputs::Dict)

Complex tensors are encoded as `<name>_real` + `<name>_imag` pairs and are
reconstructed into ComplexF32 arrays by `load_io_pair`.
"""

module PhasorTorchLoad

using HDF5

export load_params, load_io_pair, load_meta

# -------------------------------------------------------------------------
# Parameter loading: HDF5 group -> NamedTuple keyed by dataset name.
# Order matches the Python parameter_dict() ordering when possible.
# -------------------------------------------------------------------------

function _load_group_to_namedtuple(g::HDF5.Group)
    # Use a permissive Any-typed Pair list so we can mix leaf datasets
    # (Float32 arrays) with sub-groups (nested NamedTuples) freely.
    pairs = Pair{Symbol,Any}[]
    for name in keys(g)
        obj = g[name]
        if isa(obj, HDF5.Dataset)
            arr = Array{Float32}(read(obj))
            push!(pairs, Symbol(name) => arr)
        else
            # Nested layer (e.g. attn/q_proj). Recurse.
            push!(pairs, Symbol(name) => _load_group_to_namedtuple(obj))
        end
    end
    return NamedTuple(pairs)
end

"""
    load_params(path::AbstractString, layer_path::AbstractString) -> NamedTuple

Open the HDF5 file at `path`, descend to the group at `layer_path`, and
return its contents as a NamedTuple. Datasets become Float32 arrays;
sub-groups become nested NamedTuples.
"""
function load_params(path::AbstractString, layer_path::AbstractString)
    return h5open(path, "r") do f
        haskey(f, layer_path) ||
            error("missing group '$(layer_path)' in $path")
        _load_group_to_namedtuple(f[layer_path])
    end
end

# -------------------------------------------------------------------------
# IO-pair loading: (inputs, outputs) dicts.
# Complex tensors are stored as <name>_real + <name>_imag.
# -------------------------------------------------------------------------

function _load_io_group(g::HDF5.Group)
    out = Dict{String,Array}()
    names = collect(keys(g))
    seen = Set{String}()
    for name in names
        name in seen && continue
        if endswith(name, "_real")
            base = name[1:end-length("_real")]
            imag_name = base * "_imag"
            if imag_name in names
                re = Array{Float32}(read(g[name]))
                im = Array{Float32}(read(g[imag_name]))
                out[base] = ComplexF32.(re, im)
                push!(seen, name, imag_name)
                continue
            end
        end
        if endswith(name, "_imag")
            base = name[1:end-length("_imag")]
            if (base * "_real") in names
                push!(seen, name)
                continue
            end
        end
        out[name] = Array{Float32}(read(g[name]))
        push!(seen, name)
    end
    return out
end

"""
    load_io_pair(path) -> (inputs::Dict{String,Array}, outputs::Dict{String,Array})

Read the `inputs/` and `outputs/` groups from an IO-pair HDF5 file.
"""
function load_io_pair(path::AbstractString)
    return h5open(path, "r") do f
        inputs = _load_io_group(f["inputs"])
        outputs = _load_io_group(f["outputs"])
        (inputs, outputs)
    end
end

"""
    load_meta(path) -> Dict{String,String}

Read the root-level string attributes from an HDF5 file (e.g. metadata
attached by `save_state` / `save_io_pair`).
"""
function load_meta(path::AbstractString)
    return h5open(path, "r") do f
        out = Dict{String,String}()
        for k in keys(attributes(f))
            v = read_attribute(f, k)
            out[k] = string(v)
        end
        out
    end
end

end # module
