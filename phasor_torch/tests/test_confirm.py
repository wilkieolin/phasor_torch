"""Tests for the full-data confirmation runner (phasor_torch.confirm)."""

from phasor_torch import confirm, hpo


def test_read_top_orders_best_first(tmp_path):
    csvp = tmp_path / "results.csv"
    csvp.write_text(
        "d_hidden_i,epochs,init_scale,lr,n_anchors_i,n_heads_i,readout_frac,weight_decay,objective\n"
        "0,5,3.0,3e-4,0,1,0.25,1e-8,-0.10\n"
        "1,5,3.0,3e-4,1,1,0.25,1e-8,-0.50\n"   # best (most negative)
        "2,5,3.0,3e-4,2,1,0.25,1e-8,-0.30\n"
    )
    top = confirm.read_top(str(csvp), 2)
    assert [r["objective"] for r in top] == ["-0.50", "-0.30"]


def test_point_from_row_drops_objective_and_maps_full_data():
    base = hpo.HpoBase(body="lca", source="audio",
                       train_path="/x/tr.h5", test_path="/x/te.h5")  # no limits = full data
    row = {"d_hidden_i": "2", "epochs": "7", "init_scale": "2.0", "lr": "5e-4",
           "n_anchors_i": "3", "n_heads_i": "2", "readout_frac": "0.2",
           "weight_decay": "1e-7", "objective": "-0.5"}
    point = confirm._point_from_row(row)
    assert "objective" not in point
    run = hpo.point_to_runconfig(point, base)
    # index dims resolve via DISCRETE_CHOICES; strings coerce fine
    assert run.model.d_hidden == 256 and run.model.n_anchors == 256 and run.model.n_heads == 8
    assert run.train.epochs == 7 and run.train.lr == 5e-4
    assert run.data.train_limit is None and run.data.test_limit is None   # full data
