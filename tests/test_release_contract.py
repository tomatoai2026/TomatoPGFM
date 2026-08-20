import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_training_snapshot():
    config = json.loads((ROOT / "configs/pretraining_final.json").read_text(encoding="utf-8"))
    assert config["total_steps"] == 116_883
    assert [stage["steps"] for stage in config["stages"]] == [52_684, 32_925, 16_460, 9_876, 4_938]
    assert config["weights"] == {
        "mlm": 1.0,
        "vtp": 0.0,
        "cpc": 0.0,
        "masked_path": 0.25,
        "graph_recon": 0.25,
    }


def test_external_manifests_have_coordinates():
    expected = {"LA1974": 17_228, "MicroTom": 23_504}
    for genome, expected_rows in expected.items():
        path = ROOT / "manifests" / f"{genome}_evaluation_windows_coordinates_only.jsonl.gz"
        rows = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                assert row["coordinate_system"] == "0-based half-open"
                assert row["end"] - row["start"] == 512
                assert "seq" not in row
                assert len(row["sequence_sha256"]) == 64
        assert rows == expected_rows


def test_public_training_panel_is_path_free():
    path = ROOT / "manifests" / "training_panel_66_public.tsv"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 67
    header = lines[0].split("\t")
    assert "fa_path" not in header
    assert "gff_path" not in header
    assert all("C:\\Users\\" not in line and "/data/" not in line for line in lines)


def test_reported_padding_aware_auroc():
    expected = {
        "LA1974": {"gene_vs_intergenic": 0.8638200628208723, "cds_vs_intergenic": 0.959189050093128},
        "MicroTom": {"gene_vs_intergenic": 0.8489196253918477, "cds_vs_intergenic": 0.9593430745079097},
    }
    for genome, tasks in expected.items():
        result = json.loads(
            (ROOT / "results/frozen_probe" / f"{genome}.json").read_text(encoding="utf-8")
        )
        for task, auroc in tasks.items():
            assert result["tasks"][task]["models"]["TomatoPGFM_masked"]["auroc"] == auroc
