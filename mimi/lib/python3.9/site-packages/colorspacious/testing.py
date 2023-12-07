# This file is part of colorspacious
# Copyright (C) 2014-2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np

# test_vector is like
#   [(gold_a, gold_b),
#    (gold_a, gold_b),
#    ...
#   ]

def check_conversion(forward, backward, test_vector,
                     a_min=0, a_max=100,
                     b_min=0, b_max=100,
                     # We use different precisions for tests against gold
                     # standard test vectors collected elsewhere, versus
                     # checks of our internal consistency. This is because
                     # other people's codebases introduce all kinds of
                     # rounding error -- they precompute inverse matrices and
                     # then round them before using, they replace fractions
                     # with decimals rounded to like 4 significant digits,
                     # etc. Internally, though, we strive for higher
                     # standards.
                     gold_rtol=1e-3, internal_rtol=1e-8,
                     **kwargs):
    a_min = np.asarray(a_min)
    a_max = np.asarray(a_max)
    b_min = np.asarray(b_min)
    b_max = np.asarray(b_max)

    def check_one(one_a, one_b):
        conv_b = forward(one_a, **kwargs)
        assert np.allclose(conv_b, one_b, rtol=gold_rtol)
        conv_a = backward(one_b, **kwargs)
        assert np.allclose(conv_a, one_a, rtol=gold_rtol)

    for a, b in test_vector:
        check_one(a, b)

    all_a = np.asarray([pair[0] for pair in test_vector])
    all_b = np.asarray([pair[1] for pair in test_vector])
    check_one(all_a, all_b)

    # make sure broadcasting / high-dimensionality conversion works
    all_a_3d = all_a[np.newaxis, ...]
    all_b_3d = all_b[np.newaxis, ...]
    check_one(all_a_3d, all_b_3d)

    # We use a stricter tolerance here, because
    r = np.random.RandomState(0)
    rand_a = r.uniform(size=(2, 2, 2, 2, 3))
    rand_a *= (a_max - a_min)
    rand_a += a_min
    assert np.allclose(backward(forward(rand_a, **kwargs), **kwargs),
                       rand_a,
                       rtol=internal_rtol)
    rand_b = r.uniform(size=(2, 2, 2, 2, 3))
    rand_b *= (b_max - b_min)
    rand_b += b_min
    assert np.allclose(forward(backward(rand_b, **kwargs), **kwargs),
                       rand_b,
                       rtol=internal_rtol)
