"""Build the fixed Tier 3 test set (seeded, saved to data/test_set/)."""
from gwdetector.simulate import SimConfig
from gwdetector.testset import build_test_set

if __name__ == "__main__":
    d = build_test_set(SimConfig())
    size = sum(f.stat().st_size for f in d.iterdir()) / 1e6
    print(f"saved {d} ({size:.0f} MB)")
