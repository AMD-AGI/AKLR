# SPDX-License-Identifier: MIT
"""Frontiers and the whole-arm week-over-week rule."""

from __future__ import annotations

import copy
import unittest

from agentx_ci import pareto


def point(arm, conc, tput, intvty, sha="s", status="ok"):
    return {
        "arm": arm,
        "conc": conc,
        "status": status,
        "tput_per_gpu": tput,
        "p90_e2e_norm_intvty": intvty,
        "serve_spec_sha256": f"{sha}{conc}",
    }


def results(points, arm_hashes=None, mode="mock", aklr="a" * 40, ix="i" * 40):
    arms = sorted({p["arm"] for p in points})
    hashes = arm_hashes or {arm: f"h-{arm}" for arm in arms}
    return {
        "run": {
            "id": "1",
            "mode": mode,
            "aklr_sha": aklr,
            "image": {"run": "img"},
            "inferencex": {"sha": ix},
        },
        "serving": {
            "grid_sha256": "g",
            "arms": {a: {"arm_serve_sha256": hashes[a]} for a in arms},
        },
        "points": points,
    }


class FrontierTests(unittest.TestCase):
    def test_dominated_points_drop(self):
        pts = [
            point("a", 1, 10, 100),
            point("a", 2, 20, 90),
            point("a", 3, 15, 80),
            point("a", 4, 30, 50),
        ]
        self.assertEqual([p["conc"] for p in pareto.frontier(pts)], [4, 2, 1])

    def test_failed_points_are_not_measured(self):
        r = results([point("a", 1, 10, 100), point("a", 2, 99, 99, status="failed")])
        self.assertEqual([p["conc"] for p in pareto.frontiers(r)["arms"]["a"]], [1])

    def test_combined_frontier_spans_arms(self):
        r = results(
            [
                point("tp8", 1, 20, 140),
                point("tp8", 4, 70, 105),
                point("tp4", 4, 90, 80),
                point("tp4", 16, 280, 60),
            ]
        )
        combined = pareto.frontiers(r)["combined"]
        self.assertEqual(
            [(p["arm"], p["conc"]) for p in combined],
            [("tp4", 16), ("tp4", 4), ("tp8", 4), ("tp8", 1)],
        )


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.base = results(
            [
                point("tp8", 1, 20, 140),
                point("tp8", 2, 40, 120),
                point("tp4", 4, 90, 80),
            ]
        )

    def test_no_baseline(self):
        self.assertEqual(pareto.compare(self.base, None)["status"], "no-baseline")

    def test_mock_and_live_never_compare(self):
        live = copy.deepcopy(self.base)
        live["run"]["mode"] = "live"
        view = pareto.compare(live, self.base)
        self.assertEqual(view["status"], "no-baseline")
        self.assertIn("mock", view["reason"])

    def test_same_arm_hash_compares_per_point(self):
        current = results(
            [
                point("tp8", 1, 22, 140),
                point("tp8", 2, 40, 132),
                point("tp4", 4, 90, 80),
            ]
        )
        view = pareto.compare(current, self.base)["arms"]["tp8"]
        self.assertEqual(view["status"], "comparable")
        self.assertAlmostEqual(view["points"][0]["tput_ratio"], 1.1)
        self.assertAlmostEqual(view["points"][1]["intvty_ratio"], 1.1)

    def test_any_point_change_starts_a_new_segment_for_the_whole_arm(self):
        pts = [
            point("tp8", 1, 22, 140),
            point("tp8", 2, 40, 120, sha="changed"),
            point("tp4", 4, 90, 80),
        ]
        current = results(pts, arm_hashes={"tp8": "h-new", "tp4": "h-tp4"})
        view = pareto.compare(current, self.base)
        self.assertEqual(view["arms"]["tp8"]["status"], "new-segment")
        self.assertEqual(view["arms"]["tp8"]["changed_concs"], [2])
        self.assertEqual(view["arms"]["tp8"]["points"], [])
        self.assertEqual(view["arms"]["tp4"]["status"], "comparable")

    def test_new_arm_and_environment_notes(self):
        pts = [
            point("tp8", 1, 20, 140),
            point("tp8", 2, 40, 120),
            point("tp4", 4, 90, 80),
            point("tp2", 1, 5, 150),
        ]
        current = results(pts, ix="j" * 40)
        current["run"]["image"]["run"] = "img2"
        view = pareto.compare(current, self.base)
        self.assertEqual(view["arms"]["tp2"]["status"], "new-arm")
        self.assertEqual(len(view["notes"]), 2)

    def test_a_failed_point_has_no_ratio(self):
        current = results(
            [
                point("tp8", 1, 0, 0, status="failed"),
                point("tp8", 2, 40, 120),
                point("tp4", 4, 90, 80),
            ]
        )
        row = pareto.compare(current, self.base)["arms"]["tp8"]["points"][0]
        self.assertNotIn("tput_ratio", row)


if __name__ == "__main__":
    unittest.main()
