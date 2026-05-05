import numpy as np

from aida.parameter import Parameter


def test_gain_fields_accept_single_value_arrays_from_hdf5():
    parameter = Parameter(
        kx=np.array([0.95]),
        kv=np.array([0.7]),
        kT=np.array([0.95]),
        k_umin=np.array([2.0]),
        k_uk=np.array([3.0]),
    )

    assert np.allclose(parameter.kx, [0.95])
    assert np.allclose(parameter.kv, [0.7])
    assert np.allclose(parameter.kT, [0.95])
    assert np.allclose(parameter.k_umin, [2.0])
    assert np.allclose(parameter.k_uk, [3.0])
