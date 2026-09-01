import unittest

import networkx as nx

from resilience.spectral import algebraic_connectivity, edge_sensitivity, fiedler_data, relative_drop


class SpectralTests(unittest.TestCase):
    def test_disconnected_graph_has_zero_connectivity(self):
        graph = nx.Graph([(0, 1), (2, 3)])
        self.assertAlmostEqual(algebraic_connectivity(graph), 0.0)

    def test_complete_graph_connectivity(self):
        graph = nx.complete_graph(5)
        nx.set_edge_attributes(graph, 1.0, "weight")
        self.assertAlmostEqual(algebraic_connectivity(graph), 5.0)

    def test_sensitivity_is_nonnegative(self):
        graph = nx.path_graph(5)
        nx.set_edge_attributes(graph, 1.0, "weight")
        _, vector = fiedler_data(graph)
        self.assertGreaterEqual(edge_sensitivity(vector, (1, 2)), 0.0)

    def test_relative_drop_is_clipped(self):
        self.assertEqual(relative_drop(2.0, 3.0), 0.0)
        self.assertEqual(relative_drop(2.0, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
