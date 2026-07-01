import numpy as np

from rcsgcn.preprocessing import BONE_PAIRS, build_12_channel


def test_build_12_channel_order_and_shape():
    joint = np.zeros((2, 3, 5, 25, 1), dtype=np.float32)
    joint[:, 0, :, 1, 0] = np.arange(5)
    output = build_12_channel(joint)
    assert output.shape == (2, 12, 5, 25, 1)
    np.testing.assert_allclose(output[:, :3], joint)
    child, parent = BONE_PAIRS[0]
    np.testing.assert_allclose(output[:, 3:6, :, child], joint[:, :, :, child] - joint[:, :, :, parent])
    np.testing.assert_allclose(output[:, 6:9, :-1], joint[:, :, 1:] - joint[:, :, :-1])
    np.testing.assert_allclose(output[:, 6:9, -1], 0)

