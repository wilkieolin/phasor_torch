"""
verify_phasor_lca.jl — PhasorLCA parity check.

Run:
  julia --project=julia_parity julia_parity/verify_phasor_lca.jl
"""

using HDF5, PhasorNetworks, Lux, Random, Statistics

include(joinpath(@__DIR__, "load_pytorch.jl"))
using .PhasorTorchLoad

const FIXTURES = joinpath(@__DIR__, "fixtures")
const TOL = 1e-5
const TOL_LONG = 5e-4
_tol_for(name) = occursin("long", name) ? TOL_LONG : TOL

function enumerate_lca_cases(dir::AbstractString)
    cases = NamedTuple[]
    for path in sort(readdir(dir; join=true))
        endswith(path, "_weights.h5") || continue
        base = replace(basename(path), "_weights.h5" => "")
        startswith(base, "lca_") || continue
        io_path = joinpath(dir, "$(base)_io.h5")
        isfile(io_path) || continue
        push!(cases, (; name = base, weights_path = path, io_path = io_path))
    end
    return cases
end

function build_layer(meta)
    in_d = parse(Int, meta["in_dims"])
    d_model = parse(Int, meta["d_model"])
    n_heads = parse(Int, meta["n_heads"])
    n_anchors = parse(Int, meta["n_anchors"])
    init_mode = Symbol(meta["init_mode"])
    init_scale = parse(Float32, meta["init_scale"])
    t_period = parse(Float32, meta["t_period"])
    spk = SpikingArgs(t_period = t_period)
    return PhasorLCA(in_d => d_model, n_heads, n_anchors, normalize_to_unit_circle;
                     init_scale = init_scale, init_mode = init_mode, spk_args = spk)
end

function inject_params(raw::NamedTuple)
    _proj(name::Symbol) = (
        weight = permutedims(getproperty(raw, name).weight, (2, 1)),
        log_neg_lambda = vec(getproperty(raw, name).log_neg_lambda),
    )
    # anchors saved as (d_model, n_anchors) in PyTorch -> Julia HDF5 read as
    # (n_anchors, d_model) -> transpose to (d_model, n_anchors). Then wrap as Phase.
    anchors = Phase.(permutedims(raw.anchors, (2, 1)))
    return (k_proj = _proj(:k_proj),
            v_proj = _proj(:v_proj),
            anchors = anchors,
            scale = vec(raw.scale))
end

function verify_case(case; tol=_tol_for(case.name))
    meta = load_meta(case.weights_path)
    mode = meta["mode"]
    layer = build_layer(meta)
    raw = load_params(case.weights_path, "attn")
    ps = inject_params(raw)
    st = Lux.initialstates(Random.default_rng(), layer)

    inputs, outputs = load_io_pair(case.io_path)
    x_raw = inputs["x"]
    y_expected_raw = outputs["y"]

    if mode == "2d"
        x = permutedims(x_raw, (2, 1))                          # (in, B)
        y_expected = permutedims(y_expected_raw, (2, 1))        # (out, B)
    else
        x = permutedims(x_raw, (3, 2, 1))                       # (in, L, B)
        y_expected = permutedims(y_expected_raw, (3, 2, 1))     # (out, L, B)
    end
    x_phase = Phase.(x)

    y_got, _ = layer(x_phase, ps, st)
    y_got_f32 = Float32.(y_got)
    err = maximum(abs.(y_got_f32 .- y_expected))

    status = err <= tol ? "PASS" : "FAIL"
    @info "case=$(case.name) mode=$mode max|err|=$(round(err, sigdigits=5)) ($status)"
    return err
end

function main()
    cases = enumerate_lca_cases(FIXTURES)
    isempty(cases) && error("no LCA fixtures in $FIXTURES -- run generate_parity_phasor_lca.py first")
    failures = String[]
    for case in cases
        tol = _tol_for(case.name)
        err = verify_case(case; tol=tol)
        if err > tol
            push!(failures, "$(case.name): max|err|=$err > tol=$tol")
        end
    end
    if !isempty(failures)
        for f in failures
            @error f
        end
        error("PhasorLCA parity FAILED for $(length(failures)) case(s).")
    end
    @info "PhasorLCA parity: ALL CASES PASS"
end

isinteractive() || main()
