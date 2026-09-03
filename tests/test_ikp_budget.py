import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
budget = importlib.import_module("ikp_budget")


class BudgetProbeCountTests(unittest.TestCase):
    def test_sample_count_matches_estimator_contract(self):
        self.assertEqual(budget.resolve_probe_count(10), 10)
        self.assertEqual(budget.resolve_probe_count(200), 200)
        self.assertEqual(budget.resolve_probe_count(None), budget.FULL_PROBE_COUNT)


if __name__ == "__main__":
    unittest.main()
