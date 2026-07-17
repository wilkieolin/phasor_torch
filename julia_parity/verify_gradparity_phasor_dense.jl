"""
verify_gradparity_phasor_dense.jl — PhasorDense BACKWARD parity check.

Loads fixtures from generate_gradparity_phasor_dense.py and asserts that Lux +
Zygote gradients of  loss = sum(y .* ybar)  match PyTorch autograd gradients on
the same weights + input + cotangent. This tests the training backward path
(the `_exp_kdt` rrule, causal-conv / bias-kernel backward, and the
complex_to_angle 1e-3 grad gate) that the forward-only harness never covered.

Run:  julia --project=julia_parity julia_parity/verify_gradparity_phasor_dense.jl
"""

using HDF5, PhasorNetworks, Lux, Random, Statistics, Zygote

include(joinpath(@__DIR__, "load_pytorch.jl"))
using .PhasorTorchLoad

const FIXTURES = joinpath(@__DIR__, "fixtures")
const RTOL = 2e-3          # relative L2 tolerance (grads compound; Toeplitz path)
const RTOL_LONG = 2e-2     # long-sequence FFT path

function enumerate_cases(dir)
    cases = NamedTuple[]
    for path in sort(readdir(dir; join=true))
        endswith(path, "_weights.h5") || continue
        name = replace(basename(path), "_weights.h5" => "")
        startswith(name, "gdense_") || continue
        push!(cases, (; name, weights_path = path,
                      io_path = joinpath(dir, "$(name)_io.h5"),
                      grads_path = joinpath(dir, "$(name)_grads.h5")))
    end
    return cases
end

function build_lux_layer(meta::Dict)
    PhasorDense(parse(Int, meta["in_dims"]) => parse(Int, meta["out_dims"]),
                normalize_to_unit_circle;
                use_bias = meta["use_bias"] == "True",
                init_mode = Symbol(meta["init_mode"]),
                spk_args = SpikingArgs(t_period = parse(Float32, meta["t_period"])))
end

function injected_params(layer::PhasorDense, raw::NamedTuple)
    w = permutedims(raw.weight, (2, 1))
    lnl = vec(raw.log_neg_lambda)
    layer.use_bias ?
        (weight = w, log_neg_lambda = lnl, bias_real = vec(raw.bias_real), bias_imag = vec(raw.bias_imag)) :
        (weight = w, log_neg_lambda = lnl)
end

relerr(got, exp) = sqrt(sum(abs2, got .- exp)) / (sqrt(sum(abs2, exp)) + 1f-12)

function verify_case(case)
    meta = load_meta(case.weights_path)
    layer = build_lux_layer(meta)
    ps = injected_params(layer, load_params(case.weights_path, "dense"))
    st = Lux.initialstates(Random.default_rng(), layer)

    inputs, _ = load_io_pair(case.io_path)
    x = permutedims(inputs["x"], (3, 2, 1))              # (C_in, L, B)
    ybar = permutedims(inputs["ybar"], (3, 2, 1))        # (out, L, B)
    xph = Phase.(x)

    loss(p) = sum(Float32.(layer(xph, p, st)[1]) .* ybar)
    gp = Zygote.gradient(loss, ps)[1]

    g = h5open(case.grads_path, "r") do f
        d = Dict{String,Any}()
        for k in keys(f); d[k] = read(f[k]); end
        d
    end
    # torch weight grad (out,in) row-major -> Julia (in,out) -> permute back.
    tw   = permutedims(g["weight"], (2, 1))
    tlnl = vec(g["log_neg_lambda"])

    L = size(x, 2)
    rtol = L > 64 ? RTOL_LONG : RTOL
    errs = Dict("weight" => relerr(gp.weight, tw),
                "log_neg_lambda" => relerr(gp.log_neg_lambda, tlnl))
    if layer.use_bias
        errs["bias_real"] = relerr(gp.bias_real, vec(g["bias_real"]))
        errs["bias_imag"] = relerr(gp.bias_imag, vec(g["bias_imag"]))
    end
    worst = maximum(values(errs))
    status = worst <= rtol ? "PASS" : "FAIL"
    detail = join(["$k=$(round(v, sigdigits=3))" for (k, v) in sort(collect(errs))], "  ")
    @info "case=$(case.name) L=$L rtol=$rtol  $detail  ($status)"
    return worst, rtol
end

function main()
    isdir(FIXTURES) || error("fixtures not found; run generate_gradparity_phasor_dense.py first")
    cases = enumerate_cases(FIXTURES)
    isempty(cases) && error("no gdense_ gradient cases found")
    fails = String[]
    for case in cases
        worst, rtol = verify_case(case)
        worst > rtol && push!(fails, "$(case.name): relerr=$worst > $rtol")
    end
    if !isempty(fails)
        for f in fails; @error f; end
        error("PhasorDense GRADIENT parity FAILED for $(length(fails)) case(s).")
    end
    @info "PhasorDense GRADIENT parity: ALL CASES PASS"
end

isinteractive() || main()
