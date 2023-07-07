def format_coord(self, theta, r):
     # docstring inherited 
    screen_xy = self.transData.transform((theta, r))
    screen_xys = screen_xy + np.stack(
        np.meshgrid([-1, 0, 1], [-1, 0, 1])).reshape((2, -1)).T
    ts, rs = self.transData.inverted().transform(screen_xys).T
    delta_t = abs((ts - theta + np.pi) % (2 * np.pi) - np.pi).max()
    delta_t_halfturns = delta_t / np.pi
    delta_t_degrees = delta_t_halfturns * 180
    delta_r = abs(rs - r).max()
    if theta < 0:
        theta += 2 * np.pi
    theta_halfturns = theta / np.pi
    theta_degrees = theta_halfturns * 180

    # See ScalarFormatter.format_data_short. For r, use #g-formatting
    # (as for linear axes), but for theta, use f-formatting as scientific
    # notation doesn't make sense and the trailing dot is ugly.
    def format_sig(value, delta, opt, fmt):
        # For "f", only count digits after decimal point.
        prec = (max(0, -math.floor(math.log10(delta))) if fmt == "f" else
                cbook._g_sig_digits(value, delta))
        return f"{value:-{opt}.{prec}{fmt}}"
    #retrieves corresponding formatting information to properly plot on the polar plot.
    fmt_theta = self.xaxis.get_major_formatter().format_data
    fmt_r = self.yaxis.get_major_formatter().format_data

    return ('\N{GREEK SMALL LETTER THETA}={}\N{GREEK SMALL LETTER PI} '
            '({}\N{DEGREE SIGN}), r={}').format(
                format_sig(fmt_theta(theta), delta_t_halfturns, "", "f"),
                format_sig(theta_degrees, delta_t_degrees, "", "f"),
                format_sig(fmt_r(r), delta_r, "#", "g"),
            )
