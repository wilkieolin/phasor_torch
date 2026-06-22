"""
verify_audio_e2e.jl — end-to-end AUDIO chain parity from a trained checkpoint.

Loads the weights + IO pair produced by
`generate_audio_e2e_from_checkpoint.py`, rebuilds the full audio chain in Lux
(encode_input -> ResonantSTFT -> downsample -> to_phase -> PhasorDense ->
PhasorLCA -> PhasorDense -> SSMReadout), injects the trained PyTorch weights,
runs forward on the same real-audio batch, and compares the similarity scores
and argmax predictions against PyTorch.

Run:
  julia --project=julia_parity julia_parity/verify_audio_e2e.jl
"""

using HDF5, PhasorNetworks, Lux, Random, Statistics

include(joinpath(@__DIR__, "load_pytorch.jl"))
using .PhasorTorchLoad

const FIXTURES = joinpath(@__DIR__, "fixtures")
const WEIGHTS = joinpath(FIXTURES, "audio_e2e_weights.h5")
const IO = joinpath(FIXTURES, "audio_e2e_io.h5")
# Full chain at L=16000 (FFT conv path) through frontend + 3 dense + LCA +
# readout at float32 compounds to a few % on scores in [-1,1] (the FFT path is
# ~5e-4/layer per CLAUDE.md; this stacks five conv layers incl. the 16000-pt
# frontend). The LOAD-BEARING check is that the argmax predictions match
# PyTorch exactly; the score tolerance is a secondary numerical guard.
const SCORES_TOL = 5e-2

# Mean-pool the time axis by `ds`, matching PyTorch downsample_time
# (consecutive windows; drop the trailing remainder).
function _downsample(z::AbstractArray{<:Complex,3}, ds::Int)
    nf, L, B = size(z)
    L2 = div(L, ds)
    r = reshape(z[:, 1:L2*ds, :], nf, ds, L2, B)
    return dropdims(sum(r; dims = 2) ./ ds, dims = 2)
end

function main()
    isfile(WEIGHTS) || error("missing $WEIGHTS -- run generate_audio_e2e_from_checkpoint.py")
    meta = load_meta(IO)
    n_classes = parse(Int, meta["n_classes"])
    d_hidden  = parse(Int, meta["d_hidden"])
    n_heads   = parse(Int, meta["n_heads"])
    n_anchors = parse(Int, meta["n_anchors"])
    n_freqs   = parse(Int, meta["n_freqs"])
    ds        = parse(Int, meta["downsample_factor"])
    rfrac     = parse(Float32, meta["readout_frac"])
    init_mode = Symbol(meta["init_mode"])
    t_period  = parse(Float32, meta["t_period"])
    spk = SpikingArgs(t_period = t_period)
    rng = Random.default_rng()

    # ---- build layers (params overridden by the trained weights below) ----
    rstft = ResonantSTFT(1 => n_freqs, nothing; use_bias = false, spk_args = spk)
    l_in  = PhasorDense(n_freqs => d_hidden, normalize_to_unit_circle;
                        use_bias = false, init_mode = init_mode, spk_args = spk)
    l_body = PhasorLCA(d_hidden => d_hidden, n_heads, n_anchors, normalize_to_unit_circle;
                       init_mode = init_mode, spk_args = spk)
    l_dense = PhasorDense(d_hidden => d_hidden, identity;
                          use_bias = false, init_mode = init_mode, spk_args = spk)
    l_ro = SSMReadout(d_hidden => n_classes; readout_frac = rfrac)

    # ---- inject trained weights ----
    rf = load_params(WEIGHTS, "frontend")
    ps_f = (weight = permutedims(rf.weight, (2, 1)),
            log_neg_lambda = vec(rf.log_neg_lambda),
            omega = vec(rf.omega),
            log_r_lo = vec(rf.log_r_lo),
            log_r_gap = vec(rf.log_r_gap))

    _dense(g) = (weight = permutedims(load_params(WEIGHTS, g).weight, (2, 1)),
                 log_neg_lambda = vec(load_params(WEIGHTS, g).log_neg_lambda))
    ps_in = _dense("input")
    ps_dense = _dense("dense")

    rb = load_params(WEIGHTS, "body")
    _proj(s) = (weight = permutedims(getproperty(rb, s).weight, (2, 1)),
                log_neg_lambda = vec(getproperty(rb, s).log_neg_lambda))
    ps_body = (k_proj = _proj(:k_proj), v_proj = _proj(:v_proj),
               anchors = Phase.(permutedims(rb.anchors, (2, 1))),
               scale = vec(rb.scale))

    st_ro = (codes = Phase.(permutedims(load_params(WEIGHTS, "readout").codes, (2, 1))),)

    # ---- inputs ----
    inputs, outputs = load_io_pair(IO)
    x = permutedims(inputs["x"], (3, 2, 1))           # (1, L, B) real
    labels = Int.(round.(inputs["labels"]))           # (B,)
    sims_expected = permutedims(outputs["sims"], (2, 1))  # (n_classes, B)

    # ---- forward (mirror forward_model's frontend glue) ----
    xc = ComplexF32.(x)                                # encode_input
    z, _ = rstft(xc, ps_f, Lux.initialstates(rng, rstft))
    z = _downsample(z, ds)
    p = complex_to_angle(normalize_to_unit_circle(z))  # to_phase -> Phase
    y1, _ = l_in(p,  ps_in,   Lux.initialstates(rng, l_in))
    y2, _ = l_body(y1, ps_body, Lux.initialstates(rng, l_body))
    y3, _ = l_dense(y2, ps_dense, Lux.initialstates(rng, l_dense))
    sims_got, _ = l_ro(y3, NamedTuple(), st_ro)        # (n_classes, B)

    sims_got = Float32.(sims_got)
    score_err = maximum(abs.(sims_got .- sims_expected))
    jl_preds = vec([argmax(view(sims_got, :, i)) - 1 for i in axes(sims_got, 2)])
    pt_preds = vec([argmax(view(sims_expected, :, i)) - 1 for i in axes(sims_expected, 2)])
    pred_match = sum(jl_preds .== pt_preds)
    n = length(jl_preds)

    @info "trial=$(meta["trial"]) ckpt=$(meta["ckpt"]) body=$(meta["body"])"
    @info "max|err| on similarity scores = $(round(score_err, sigdigits=5)) (tol=$SCORES_TOL)"
    @info "argmax predictions match: $pred_match / $n"
    @info "  Julia preds:   $jl_preds"
    @info "  PyTorch preds: $pt_preds"

    if pred_match != n
        error("argmax mismatch: $(n - pred_match)/$n samples differ -- weights are NOT equivalent")
    end
    if score_err > SCORES_TOL
        error("score parity failed: max|err|=$score_err > $SCORES_TOL")
    end
    @info "AUDIO end-to-end parity: PASS (predictions match, scores within tol)"
end

isinteractive() || main()
