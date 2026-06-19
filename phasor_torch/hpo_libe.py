"""libEnsemble launcher for the ytopt HPO — multi-node / multi-tile scale-out.

This is the parallel HPC path (ytopt-libe pattern), replacing the Ray evaluator
which does not work on PBS nodes. libEnsemble runs a persistent ytopt generator
(ask/tell) on one worker and one trial per remaining worker; under MPI each
worker is a rank, so trials run one-per-rank and can be pinned one-per-XPU-tile.

It reuses the single-source-of-truth driver in `phasor_torch.hpo`:
`make_space` (search space) and `objective` (train one config -> -test_acc +
per-trial artifacts). The generator/simulator wrappers below follow the
canonical ytopt-libe `persistent_ytopt` / `init_obj` templates.

Launch (inside the `phasor_hpo` env; PHASOR_HPO_* configure the study):
    # single node, local comms (smoke / one node):
    python -m phasor_torch.hpo_libe --comms local --nworkers 4 \\
        --learner RF --max-evals 50
    # multi-node under PBS (one rank per tile, manager+gen on two of them):
    mpiexec -np <ranks> ... python -m phasor_torch.hpo_libe --learner RF --max-evals 200

With N workers, one is the generator and N-1 run trials concurrently.
"""

from __future__ import annotations

import os
import secrets
import time

import numpy as np

from . import hpo


# --------------------------------------------------------------------------
# Generator (gen_f): persistent ytopt ask/tell. Generic over the field set;
# mirrors the canonical ytopt-libe persistent_ytopt.
# --------------------------------------------------------------------------


def persistent_ytopt(H, persis_info, gen_specs, libE_info):
    from libensemble.message_numbers import EVAL_GEN_TAG, PERSIS_STOP, STOP_TAG
    from libensemble.tools.persistent_support import PersistentSupport

    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    user = gen_specs["user"]
    ytoptimizer = user["ytoptimizer"]
    num_sim_workers = user["num_sim_workers"]

    fields = [f[0] for f in gen_specs["out"]]
    tag = None
    calc_in = None
    first_call = True

    while tag not in [STOP_TAG, PERSIS_STOP]:
        if first_call:
            ytopt_points = ytoptimizer.ask_initial(n_points=num_sim_workers)
            batch_size = len(ytopt_points)
            first_call = False
        else:
            batch_size = len(calc_in)
            results = []
            for entry in calc_in:
                params = {f: entry[f][0] for f in fields}
                results.append((params, entry["objective"]))
            ytoptimizer.tell(results)
            ytopt_points = list(ytoptimizer.ask(n_points=batch_size))[0]

        H_o = np.zeros(batch_size, dtype=gen_specs["out"])
        for i, entry in enumerate(ytopt_points):
            for key, value in entry.items():
                H_o[i][key] = value

        tag, Work, calc_in = ps.send_recv(H_o)

    return H_o, persis_info, _finished_tag()


def _finished_tag():
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    return FINISHED_PERSISTENT_GEN_TAG


# --------------------------------------------------------------------------
# Simulator (sim_f): build the point dict and call the shared objective.
# --------------------------------------------------------------------------


def init_obj(H, persis_info, sim_specs, libE_info):
    point = {f: np.squeeze(H[f]).item() for f in sim_specs["in"]}
    y = hpo.objective(point)                       # trains + writes per-trial artifacts
    H_o = np.zeros(1, dtype=sim_specs["out"])
    H_o["objective"] = y
    H_o["elapsed_sec"] = time.time() - _START
    return H_o, persis_info


_START = time.time()


# --------------------------------------------------------------------------
# Field <-> numpy dtype mapping derived from the ConfigSpace.
# --------------------------------------------------------------------------


def _field_specs(cs) -> list[tuple[str, type]]:
    """Return [(name, np_type)] for each hyperparameter, in space order.

    Floats -> float, integers and integer-choice categoricals -> int. (All of
    our categoricals — d_hidden/n_heads/n_anchors — have int choices.)
    """
    import ConfigSpace.hyperparameters as CSH

    names = list(cs.keys()) if hasattr(cs, "keys") else cs.get_hyperparameter_names()
    out: list[tuple[str, type]] = []
    for name in names:
        hp = cs[name] if hasattr(cs, "__getitem__") else cs.get_hyperparameter(name)
        if isinstance(hp, CSH.UniformFloatHyperparameter):
            out.append((name, float))
        elif isinstance(hp, CSH.UniformIntegerHyperparameter):
            out.append((name, int))
        elif isinstance(hp, (CSH.CategoricalHyperparameter, CSH.OrdinalHyperparameter)):
            choices = hp.choices if hasattr(hp, "choices") else hp.sequence
            is_int = all(isinstance(c, (int, np.integer)) for c in choices)
            out.append((name, int if is_int else "<U24"))
        else:
            out.append((name, float))
    return out


def _parse_user_args(user_args_in: list[str]) -> dict:
    user_args: dict[str, str] = {}
    for entry in user_args_in:
        if entry.startswith("--"):
            if "=" in entry:
                k, v = entry[2:].split("=", 1)
            else:
                k = entry[2:]
                v = user_args_in[user_args_in.index(entry) + 1]
            user_args[k] = v
    return user_args


# --------------------------------------------------------------------------
# Calling script.
# --------------------------------------------------------------------------


def main():
    from libensemble.alloc_funcs.start_only_persistent import (
        only_persistent_gens as alloc_f,
    )
    from libensemble.libE import libE
    from libensemble.tools import add_unique_random_streams, parse_args
    from ytopt.search.optimizer import Optimizer

    nworkers, is_manager, libE_specs, user_args_in = parse_args()
    num_sim_workers = nworkers - 1  # one worker hosts the persistent generator

    user_args = _parse_user_args(user_args_in)
    for req in ("learner", "max-evals"):
        assert req in user_args, f"missing --{req} (e.g. --learner RF --max-evals 50)"

    base = hpo.HpoBase.from_env()
    cs = hpo.make_space(base)
    fields = _field_specs(cs)
    field_names = [n for n, _ in fields]
    results_path = os.path.abspath(
        os.environ.get("PHASOR_HPO_RESULTS",
                       os.path.join(base.outdir, "results.csv"))
    )

    sim_specs = {
        "sim_f": init_obj,
        "in": field_names,
        "out": [("objective", float), ("elapsed_sec", float)],
    }
    gen_specs = {
        "gen_f": persistent_ytopt,
        "out": [(n, t, (1,)) for n, t in fields],
        "persis_in": field_names + ["objective", "elapsed_sec"],
        "user": {
            "ytoptimizer": Optimizer(
                num_workers=num_sim_workers,
                space=cs,
                learner=user_args["learner"],
                liar_strategy="cl_max",
                acq_func="gp_hedge",
                set_KAPPA=1.96,
                set_SEED=2345,
                set_NI=10,
            ),
            "num_sim_workers": num_sim_workers,
        },
    }
    alloc_specs = {"alloc_f": alloc_f, "user": {"async_return": True}}
    exit_criteria = {"gen_max": int(user_args["max-evals"])}
    # Graceful stop: if PHASOR_HPO_WALLCLOCK_MAX (seconds) is set, libE stops
    # issuing new trials at that point and the manager still writes results.csv
    # below -- set it a bit under the PBS walltime so the job isn't killed
    # mid-run with no trajectory CSV.
    _wc = os.environ.get("PHASOR_HPO_WALLCLOCK_MAX")
    if _wc:
        exit_criteria["wallclock_max"] = float(_wc)

    os.makedirs(base.outdir, exist_ok=True)
    libE_specs["use_worker_dirs"] = True
    libE_specs["sim_dirs_make"] = False
    libE_specs["ensemble_dir_path"] = os.environ.get(
        "PHASOR_HPO_ENSEMBLE_DIR", "./ensemble_" + secrets.token_hex(nbytes=4)
    )
    # Allow re-running into the same ensemble dir (our per-trial artifacts are
    # keyed by point hash and our results.csv is rewritten from the full history,
    # so re-use is safe and avoids a manual rm between runs).
    libE_specs["reuse_output_dir"] = True

    persis_info = add_unique_random_streams({}, nworkers + 1)

    H, persis_info, flag = libE(
        sim_specs, gen_specs, exit_criteria, persis_info,
        alloc_specs=alloc_specs, libE_specs=libE_specs,
    )

    if is_manager:
        # Authoritative trajectory CSV from the full history (all completed sims),
        # written once here rather than incrementally from the generator.
        done = H[H["sim_ended"]] if "sim_ended" in H.dtype.names else H
        cols = field_names + ["objective"]
        with open(results_path, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for row in done:
                fh.write(",".join(str(np.squeeze(row[c])) for c in cols) + "\n")
        print(f"libEnsemble HPO complete: {len(done)} evaluations. results: {results_path}",
              flush=True)


if __name__ == "__main__":
    main()
