from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _heading_positions(text: str, headings: list[str]) -> list[int]:
    positions = []
    for heading in headings:
        pos = text.find(heading)
        assert pos != -1, f"missing heading: {heading}"
        positions.append(pos)
    return positions


def test_readme_top_fold_has_launch_contract():
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    top = "\n".join(readme.splitlines()[:40])

    assert "# FD-CATE: Personalized Causal Inference Under Unmeasured Confounding" in readme
    assert "https://arxiv.org/abs/2509.22531" in top
    assert "python -m pip install fd-cate" in top
    assert "fdcate demo --outdir ./fdcate-demo" in top
    assert "fdcate_nsweep_rho2_d30_fullnoise_plot.png" in top
    assert (
        "Estimate heterogeneous treatment effects even when treatment and outcome share hidden confounders"
        in top
    )
    assert "### Who is this for?" in readme

    positions = _heading_positions(
        readme,
        [
            "## Why it matters",
            "## One-command quickstart",
            "## What you get",
            "## Main figure",
            "## Reproduce paper",
            "## Citation",
            "## Links",
        ],
    )
    assert positions == sorted(positions)


def test_citation_cff_matches_package_version_and_paper():
    root = _repo_root()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")

    assert 'version = "0.1.1"' in pyproject
    assert 'version: 0.1.1' in citation
    assert "preferred-citation:" in citation
    assert "https://arxiv.org/abs/2509.22531" in citation
