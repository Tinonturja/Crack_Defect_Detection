import torch

from crack_detector.model import build_model, count_trainable_parameters


def test_build_model_output_shape():
    model = build_model()
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    assert output.shape == (4,)  # single logit per image


def test_build_model_works_at_different_input_sizes():
    """Thanks to AdaptiveAvgPool2d, the model isn't locked to one input size."""
    model = build_model()
    for size in (32, 64, 128):
        dummy_input = torch.randn(2, 3, size, size)
        output = model(dummy_input)
        assert output.shape == (2,)


def test_count_trainable_parameters_positive():
    model = build_model()
    assert count_trainable_parameters(model) > 0
