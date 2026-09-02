from pathlib import Path
import unittest

import networkx as nx
import torch

from resilience.model import ScenarioGCN, ScenarioGraphSAGE
from resilience.osm import load_preprocessed_network


class ExtendedModelTests(unittest.TestCase):
    def test_direct_prediction_is_bounded(self):
        model = ScenarioGCN(input_dim=5, residual=False)
        x = torch.zeros((6, 5))
        adjacency = torch.eye(6)
        value = float(model(x, adjacency).detach())
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_zero_residual_preserves_spectral_baseline(self):
        model = ScenarioGCN(input_dim=5, residual=True)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.zeros((6, 5))
        adjacency = torch.eye(6)
        baseline = torch.tensor(0.37)
        self.assertAlmostEqual(float(model(x, adjacency, baseline).detach()), 0.37, places=6)

    def test_cached_osm_network_is_connected(self):
        path = Path("data/osm/hcmus_650m_drive.graphml")
        if not path.exists():
            self.skipTest("Cached OSM fixture not present")
        graph = load_preprocessed_network(path)
        self.assertTrue(nx.is_connected(graph))
        self.assertEqual(graph.number_of_nodes(), 163)
        self.assertEqual(graph.number_of_edges(), 221)

    def test_graphsage_direct_and_residual_are_bounded(self):
        x = torch.zeros((6, 5))
        adjacency = torch.eye(6)
        direct = float(ScenarioGraphSAGE(input_dim=5)(x, adjacency).detach())
        residual = ScenarioGraphSAGE(input_dim=5, residual=True)
        for parameter in residual.parameters():
            torch.nn.init.zeros_(parameter)
        corrected = float(residual(x, adjacency, torch.tensor(0.37)).detach())
        self.assertGreaterEqual(direct, 0.0)
        self.assertLessEqual(direct, 1.0)
        self.assertAlmostEqual(corrected, 0.37, places=6)


if __name__ == "__main__":
    unittest.main()
