import os

import pytest

"""Initialization for the tests package."""

if __name__ == "__main__":
    test_directory: str = os.path.dirname(__file__)
    pytest.main([os.path.join(test_directory, "test_mdist_data_loader")])
    pytest.main([os.path.join(test_directory, "test_mdist_nn.py")])
