import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_variants import generate_all_variants, PROXY_DIMENSIONS


def test_substance_identical_within_base():
    variants = generate_all_variants(full_factorial=True)
    by_base = {}
    for v in variants:
        substance_only = v.resume_text.split("Experience:\n", 1)[-1]
        by_base.setdefault(v.base_resume_id, set()).add(substance_only)

    for base_id, substances in by_base.items():
        assert len(substances) == 1, f"Substance drift detected in {base_id}"


def test_proxy_labels_match_toggled_content():
    variants = generate_all_variants(full_factorial=True)
    for v in variants:
        if v.employment_gap == "gap_present":
            assert "Career Break" in v.resume_text
        else:
            assert "Career Break" not in v.resume_text

        if v.years_framing == "explicit_years":
            assert "years of professional" in v.resume_text
        else:
            assert "Experience demonstrated through the project" in v.resume_text


def test_all_proxy_dimensions_represented():
    variants = generate_all_variants(full_factorial=True)
    for dimension, levels in PROXY_DIMENSIONS.items():
        seen = {getattr(v, dimension) for v in variants}
        assert seen == set(levels), f"Not all levels of {dimension} were sampled: {seen}"


if __name__ == "__main__":
    test_substance_identical_within_base()
    test_proxy_labels_match_toggled_content()
    test_all_proxy_dimensions_represented()
    print("All tests passed.")
