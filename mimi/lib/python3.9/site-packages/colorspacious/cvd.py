# This file is part of colorspacious
# Copyright (C) 2015 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Simulation of color vision deficiency (a.k.a. color blindness)

import numpy as np

# Matrices for simulating anomalous color vision from:
#   Machado, Oliveira, & Fernandes (2009). A Physiologically-based Model for
#   Simulation of Color Vision Deficiency. doi: 10.1109/TVCG.2009.113
#
#   http://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html

# Most people with anomalous color vision (~5% of all men) fall somewhere on
# the deuteranomaly spectrum. A minority (~1% of all men) are either fully
# deuteranopic or fall on the protanomaly spectrum. A much smaller number fall
# on the tritanomaly spectrum (<0.01% of people) or have other more exotic
# anomalies.

def machado_et_al_2009_matrix(cvd_type, severity):
    """Retrieve a matrix for simulating anomalous color vision.

    :param cvd_type: One of "protanomaly", "deuteranomaly", or "tritanomaly".
    :param severity: A value between 0 and 100.

    :returns: A 3x3 CVD simulation matrix as computed by Machado et al
             (2009).

    These matrices were downloaded from:

      http://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html

    which is supplementary data from :cite:`Machado-CVD`.

    If severity is a multiple of 10, then simply returns the matrix from that
    webpage. For other severities, performs linear interpolation.
    """

    assert 0 <= severity <= 100

    fraction = severity % 10

    low = int(severity - fraction)
    high = low + 10
    assert low <= severity <= high

    low_matrix = np.asarray(MACHADO_ET_AL_MATRICES[cvd_type][low])
    if severity == 100:
        # Don't try interpolating between 100 and 110, there is no 110...
        return low_matrix
    high_matrix = np.asarray(MACHADO_ET_AL_MATRICES[cvd_type][high])
    return ((1 - fraction / 10.0) * low_matrix
            + fraction / 10.0 * high_matrix)

def test_machado_et_al_2009_matrix():
    np.testing.assert_almost_equal(
        machado_et_al_2009_matrix("deuteranomaly", 50),
        [[ 0.547494, 0.607765, -0.155259],
         [ 0.181692, 0.781742,  0.036566],
         [-0.010410, 0.027275,  0.983136]])

    deuter50 = np.asarray([[ 0.547494, 0.607765, -0.155259],
                           [ 0.181692, 0.781742,  0.036566],
                           [-0.010410, 0.027275,  0.983136]])
    deuter60 = np.asarray([[ 0.498864, 0.674741, -0.173604],
                           [ 0.205199, 0.754872,  0.039929],
                           [-0.011131, 0.030969,  0.980162]])
    np.testing.assert_almost_equal(
        machado_et_al_2009_matrix("deuteranomaly", 53.1),
        0.31 * deuter60 + (1 - 0.31) * deuter50)

    # Test that 0 and 100 work as arguments
    assert np.allclose(machado_et_al_2009_matrix("protanomaly", 0)[0, 0],
                       1.0)
    assert np.allclose(machado_et_al_2009_matrix("protanomaly", 100)[0, 0],
                       0.152286)

MACHADO_ET_AL_MATRICES = {
    "protanomaly": {
      0: [
          [ 1.000000,  0.000000, -0.000000],
          [ 0.000000,  1.000000,  0.000000],
          [-0.000000, -0.000000,  1.000000],
         ],
     10: [
          [ 0.856167,  0.182038, -0.038205],
          [ 0.029342,  0.955115,  0.015544],
          [-0.002880, -0.001563,  1.004443],
         ],
     20: [
          [ 0.734766,  0.334872, -0.069637],
          [ 0.051840,  0.919198,  0.028963],
          [-0.004928, -0.004209,  1.009137],
         ],
     30: [
          [ 0.630323,  0.465641, -0.095964],
          [ 0.069181,  0.890046,  0.040773],
          [-0.006308, -0.007724,  1.014032],
         ],
     40: [
          [ 0.539009,  0.579343, -0.118352],
          [ 0.082546,  0.866121,  0.051332],
          [-0.007136, -0.011959,  1.019095],
         ],
     50: [
          [ 0.458064,  0.679578, -0.137642],
          [ 0.092785,  0.846313,  0.060902],
          [-0.007494, -0.016807,  1.024301],
         ],
     60: [
          [ 0.385450,  0.769005, -0.154455],
          [ 0.100526,  0.829802,  0.069673],
          [-0.007442, -0.022190,  1.029632],
         ],
     70: [
          [ 0.319627,  0.849633, -0.169261],
          [ 0.106241,  0.815969,  0.077790],
          [-0.007025, -0.028051,  1.035076],
         ],
     80: [
          [ 0.259411,  0.923008, -0.182420],
          [ 0.110296,  0.804340,  0.085364],
          [-0.006276, -0.034346,  1.040622],
         ],
     90: [
          [ 0.203876,  0.990338, -0.194214],
          [ 0.112975,  0.794542,  0.092483],
          [-0.005222, -0.041043,  1.046265],
         ],
    100: [
          [ 0.152286,  1.052583, -0.204868],
          [ 0.114503,  0.786281,  0.099216],
          [-0.003882, -0.048116,  1.051998],
         ],
    },
    "deuteranomaly": {
      0: [
          [ 1.000000,  0.000000, -0.000000],
          [ 0.000000,  1.000000,  0.000000],
          [-0.000000, -0.000000,  1.000000],
         ],
     10: [
          [ 0.866435,  0.177704, -0.044139],
          [ 0.049567,  0.939063,  0.011370],
          [-0.003453,  0.007233,  0.996220],
         ],
     20: [
          [ 0.760729,  0.319078, -0.079807],
          [ 0.090568,  0.889315,  0.020117],
          [-0.006027,  0.013325,  0.992702],
         ],
     30: [
          [ 0.675425,  0.433850, -0.109275],
          [ 0.125303,  0.847755,  0.026942],
          [-0.007950,  0.018572,  0.989378],
         ],
     40: [
          [ 0.605511,  0.528560, -0.134071],
          [ 0.155318,  0.812366,  0.032316],
          [-0.009376,  0.023176,  0.986200],
         ],
     50: [
          [ 0.547494,  0.607765, -0.155259],
          [ 0.181692,  0.781742,  0.036566],
          [-0.010410,  0.027275,  0.983136],
         ],
     60: [
          [ 0.498864,  0.674741, -0.173604],
          [ 0.205199,  0.754872,  0.039929],
          [-0.011131,  0.030969,  0.980162],
         ],
     70: [
          [ 0.457771,  0.731899, -0.189670],
          [ 0.226409,  0.731012,  0.042579],
          [-0.011595,  0.034333,  0.977261],
         ],
     80: [
          [ 0.422823,  0.781057, -0.203881],
          [ 0.245752,  0.709602,  0.044646],
          [-0.011843,  0.037423,  0.974421],
         ],
     90: [
          [ 0.392952,  0.823610, -0.216562],
          [ 0.263559,  0.690210,  0.046232],
          [-0.011910,  0.040281,  0.971630],
         ],
    100: [
          [ 0.367322,  0.860646, -0.227968],
          [ 0.280085,  0.672501,  0.047413],
          [-0.011820,  0.042940,  0.968881],
         ],
    },

    "tritanomaly": {
      0: [
          [ 1.000000,  0.000000, -0.000000],
          [ 0.000000,  1.000000,  0.000000],
          [-0.000000, -0.000000,  1.000000],
         ],
     10: [
          [ 0.926670,  0.092514, -0.019184],
          [ 0.021191,  0.964503,  0.014306],
          [ 0.008437,  0.054813,  0.936750],
         ],
     20: [
          [ 0.895720,  0.133330, -0.029050],
          [ 0.029997,  0.945400,  0.024603],
          [ 0.013027,  0.104707,  0.882266],
         ],
     30: [
          [ 0.905871,  0.127791, -0.033662],
          [ 0.026856,  0.941251,  0.031893],
          [ 0.013410,  0.148296,  0.838294],
         ],
     40: [
          [ 0.948035,  0.089490, -0.037526],
          [ 0.014364,  0.946792,  0.038844],
          [ 0.010853,  0.193991,  0.795156],
         ],
     50: [
          [ 1.017277,  0.027029, -0.044306],
          [-0.006113,  0.958479,  0.047634],
          [ 0.006379,  0.248708,  0.744913],
         ],
     60: [
          [ 1.104996, -0.046633, -0.058363],
          [-0.032137,  0.971635,  0.060503],
          [ 0.001336,  0.317922,  0.680742],
         ],
     70: [
          [ 1.193214, -0.109812, -0.083402],
          [-0.058496,  0.979410,  0.079086],
          [-0.002346,  0.403492,  0.598854],
         ],
     80: [
          [ 1.257728, -0.139648, -0.118081],
          [-0.078003,  0.975409,  0.102594],
          [-0.003316,  0.501214,  0.502102],
         ],
     90: [
          [ 1.278864, -0.125333, -0.153531],
          [-0.084748,  0.957674,  0.127074],
          [-0.000989,  0.601151,  0.399838],
         ],
    100: [
          [ 1.255528, -0.076749, -0.178779],
          [-0.078411,  0.930809,  0.147602],
          [ 0.004733,  0.691367,  0.303900],
         ],
    }
}

################################################################

# For reference, here's the code I used to convert a copy-paste of the
# matrices from the web page at
#
#    http://www.inf.ufrgs.br/~oliveira/pubs_files/CVD_Simulation/CVD_Simulation.html
#
# into nicely formatted source code.
#
# -----
#
# matrices = {
#     "P": {},
#     "D": {},
#     "T": {},
# }
#
# def lines_to_mat(lines):
#     [line.split() for line in lines]
#
# lines = data.split("\n")
# i = 0
# while i < len(lines):
#     chunk = lines[i:i+13]
#     severity = chunk[0].strip()
#     matrices["P"][severity] = [line.split() for line in chunk[2:5]]
#     matrices["D"][severity] = [line.split() for line in chunk[6:9]]
#     matrices["T"][severity] = [line.split() for line in chunk[10:13]]
#     i += 13
#
# def format_num(n):
#     if not n.startswith("-"):
#         return " " + n
#     else:
#         return n
#
# def print_dict(name, d):
#     print("%s = {" % (name,))
#     for severity, mats in sorted(d.items()):
#         print("    %3i: [" % (int(severity[0] + severity[2]) * 10))
#         for row in mats:
#             print("          [%s, %s, %s]," % tuple([format_num(n) for n in row]))
#         print("         ],")
#     print("}\n")
#
# print_dict("protanomaly", matrices["P"])
# print_dict("deuteranomaly", matrices["D"])
# print_dict("tritanomaly", matrices["T"])
