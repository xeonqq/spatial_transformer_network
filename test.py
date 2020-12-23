from prepare_distorted_dataset import generate_affine_distorted_mnist
from spatial_transformer_network_demo import main
import unittest


class TestSpatialTransformerNetwork(unittest.TestCase):
    def setUp(self):
        generate_affine_distorted_mnist()

    def test_spatial_transfomer_network(self):
        eval_result = main(True)
        cost, accuracy = eval_result
        self.assertGreater(accuracy, 0.85)


if __name__ == "__main__":
    unittest.main()
