"""
verify_gradparity_phasor_lca.jl — PhasorLCA BACKWARD parity check.

Compares Lux+Zygote gradients of loss = sum(y .* ybar) to PyTorch autograd for
the attention path (similarity_outer_heads, anchor bundle, VSA bind, scale) — the
prime suspect for the linchpin attn-only training gap.

Run:  julia --project=julia_parity julia_parity/verify_gradparity_phasor_lca.jl
"""

using HDF5, PhasorNetworks, Lux, Random, Statistics, Zygote

include(joinpath(@__DIR__, "load_pytorch.jl"))
using .PhasorTorchLoad

const FIXTURES = joinpath(@__DIR__, "fixtures")
const RTOL = 3e-3
const RTOL_LONG = 3e-2

function enumerate_cases(dir)
    cases = NamedTuple[]
    for path in sort(readdir(dir; join=true))
        endswith(path, "_weights.h5") || continue
        base = replace(basename(path), "_weights.h5" => "")
        startswith(base, "glca_") || continue
        push!(cases, (; name = base, weights_path = path,
                      io_path = joinpath(dir, "$(base)_io.h5"),
                      grads_path = joinpath(dir, "$(base)_grads.h5")))
    end
    return cases
end

function build_layer(meta)
    PhasorLCA(parse(Int, meta["in_dims"]) => parse(Int, meta["d_model"]),
              parse(Int, meta["n_heads"]), parse(Int, meta["n_anchors"]),
              normalize_to_unit_circle;
              init_scale = parse(Float32, meta["init_scale"]),
              init_mode = Symbol(meta["init_mode"]),
              spk_args = SpikingArgs(t_period = parse(Float32, meta["t_period"])))
end

function inject_params(raw::NamedTuple)
    _proj(name) = (weight = permutedims(getproperty(raw, name).weight, (2, 1)),
                   log_neg_lambda = vec(getproperty(raw, name).log_neg_lambda))
    (k_proj = _proj(:k_proj), v_proj = _proj(:v_proj),
     anchors = Phase.(permutedims(raw.anchors, (2, 1))), scale = vec(raw.scale))
end

relerr(got, exp) = sqrt(sum(abs2, Float32.(got) .- Float32.(exp))) / (sqrt(sum(abs2, Float32.(exp))) + 1f-12)

function verify_case(case)
    meta = load_meta(case.weights_path)
    layer = build_layer(meta)
    ps = inject_params(load_params(case.weights_path, "attn"))
    st = Lux.initialstates(Random.default_rng(), layer)

    inputs, _ = load_io_pair(case.io_path)
    xph = Phase.(permutedims(inputs["x"], (3, 2, 1)))    # (in, L, B)
    ybar = permutedims(inputs["ybar"], (3, 2, 1))         # (out, L, B)

    loss(p) = sum(Float32.(layer(xph, p, st)[1]) .* ybar)
    gp = Zygote.gradient(loss, ps)[1]

    g = h5open(case.grads_path, "r") do f
        Dict(k => read(f[k]) for k in keys(f))
    end
    errs = Dict(
        "kweight" => relerr(gp.k_proj.weight, permutedims(g["kweight"], (2, 1))),
        "vweight" => relerr(gp.v_proj.weight, permutedims(g["vweight"], (2, 1))),
        "klnl"    => relerr(gp.k_proj.log_neg_lambda, vec(g["klnl"])),
        "vlnl"    => relerr(gp.v_proj.log_neg_lambda, vec(g["vlnl"])),
        "anchors" => relerr(gp.anchors, permutedims(g["anchors"], (2, 1))),
        "scale"   => relerr(gp.scale, vec(g["scale"])),
    )
    L = size(ybar, 2)
    rtol = L > 64 ? RTOL_LONG : RTOL
    worst = maximum(values(errs))
    status = worst <= rtol ? "PASS" : "FAIL"
    detail = join(["$k=$(round(v, sigdigits=3))" for (k, v) in sort(collect(errs))], "  ")
    @info "case=$(case.name) L=$L rtol=$rtol  $detail  ($status)"
    return worst, rtol
end

function main()
    cases = enumerate_cases(FIXTURES)
    isempty(cases) && error("no glca_ gradient cases -- run generate_gradparity_phasor_lca.py first")
    fails = String[]
    for case in cases
        worst, rtol = verify_case(case)
        worst > rtol && push!(fails, "$(case.name): relerr=$worst > $rtol")
    end
    if !isempty(fails)
        for f in fails; @error f; end
        error("PhasorLCA GRADIENT parity FAILED for $(length(fails)) case(s).")
    end
    @info "PhasorLCA GRADIENT parity: ALL CASES PASS"
end

isinteractive() || main()
