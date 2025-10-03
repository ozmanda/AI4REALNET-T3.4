import unittest
import torch

from src.utils.observation.RunningMeanStd import RunningMeanStd, FeatureRunningMeanStd
from src.utils.observation.normalisation import Normalisation, FlatlandNormalisation


class NormalisationTest(unittest.TestCase):
    def setUp(self):
        self.single_feature_batch = torch.tensor([[1.0, 2.0, 3.0],
                                                  [4.0, 5.0, 6.0]], dtype=torch.float32)
        self.multi_feature_batch = torch.tensor([[1.0, 2.0, 3.0],
                                                 [4.0, 5.0, 6.0]], dtype=torch.float32)
        self.scalar_batch = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        self.tree_observation = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                               2.0, 4.0, 6.0, 8.0, 10.0]], dtype=torch.float32)


    def test_value_normalisation(self):
        rms = RunningMeanStd(size=1)
        rms.update_batch(self.single_feature_batch)

        self.assertEqual(rms.count, self.single_feature_batch.size(0))

        expected_mean = torch.mean(self.single_feature_batch)
        self.assertAlmostEqual(rms.mean.item(), expected_mean.item(), places=6)

        expected_M2 = ((self.single_feature_batch - expected_mean) ** 2).sum()
        expected_var = expected_M2 / (self.single_feature_batch.size(0) + rms.eps)
        self.assertAlmostEqual(rms.var.item(), expected_var.item(), places=6)


    def test_feature_normalisation(self):
        feature_rms = FeatureRunningMeanStd(n_features=self.multi_feature_batch.size(1))
        feature_rms.update_batch(self.multi_feature_batch)

        self.assertEqual(feature_rms.count, self.multi_feature_batch.size(0))

        expected_mean = torch.mean(self.multi_feature_batch, dim=0)
        self.assertTrue(torch.allclose(feature_rms.mean, expected_mean, atol=1e-6))

        expected_M2 = ((self.multi_feature_batch - expected_mean) ** 2).sum(dim=0)
        expected_var = expected_M2 / (self.multi_feature_batch.size(0) + feature_rms.eps)
        self.assertTrue(torch.allclose(feature_rms.var, expected_var, atol=1e-6))


    def test_tensor_normalisation(self):
        normaliser = Normalisation(eps=1e-8, clip=False)
        normaliser.update_metrics(self.scalar_batch)

        normalised = normaliser.normalise(self.scalar_batch, clip=False)

        expected_mean = self.scalar_batch.mean()
        expected_var = ((self.scalar_batch - expected_mean) ** 2).sum() / (self.scalar_batch.size(0) + normaliser.rms.eps)
        expected_std = torch.sqrt(expected_var + normaliser.eps)
        expected = (self.scalar_batch - expected_mean) / (expected_std + normaliser.eps)

        self.assertTrue(torch.allclose(normalised, expected, atol=1e-6))


    def test_tree_observation_normalisation(self):
        flat_norm = FlatlandNormalisation(n_nodes=1, n_features=12, n_agents=2, eps=1e-8, clip=False)
        normalised = flat_norm.normalise(self.tree_observation.clone(), clip=False)

        self.assertEqual(normalised.shape, self.tree_observation.shape)

        distance_mean = flat_norm.distance_rms.mean
        distance_std = flat_norm.distance_rms.std + flat_norm.eps
        expected_distance = (self.tree_observation[:, :7] - distance_mean) / distance_std

        expected_agent_counts = self.tree_observation[:, 7:10] / flat_norm.n_agents
        expected_speed = self.tree_observation[:, 10:11]
        expected_last_agent_count = self.tree_observation[:, 11:12] / flat_norm.n_agents

        expected = torch.cat(
            [expected_distance, expected_agent_counts, expected_speed, expected_last_agent_count], dim=1
        )

        self.assertTrue(torch.allclose(normalised, expected, atol=1e-6))

