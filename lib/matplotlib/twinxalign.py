"""
Let the left axis and right axis be aligned 
at the specified position in twin coordinate axis
"""


def twinxalign(ax_left, ax_right, v_left, v_right):
    """
    Let the position `v_left` on `ax_left`
    and the position `v_right` on `ax_left` be aligned
    """
    left_min, left_max = ax_left.get_ybound()
    right_min, right_max = ax_right.get_ybound()
    k = (left_max-left_min) / (right_max-right_min)
    b = left_min - k * right_min
    x_right_new = k * v_right + b
    dif = x_right_new - v_left
    if dif >= 0:
        right_min_new = ((left_min-dif) - b) / k
        k_new = (left_min-v_left) / (right_min_new-v_right)
        b_new = v_left - k_new * v_right
        right_max_new = (left_max - b_new) / k_new
    else:
        right_max_new = ((left_max-dif) - b) / k
        k_new = (left_max-v_left) / (right_max_new-v_right)
        b_new = v_left - k_new * v_right
        right_min_new = (left_min - b_new) / k_new    
    def _forward(x):
        return k_new * x + b_new
    def _inverse(x):
        return (x - b_new) / k_new
    ax_right.set_ylim([right_min_new, right_max_new])
    ax_right.set_yscale('function', functions=(_forward, _inverse))
    return ax_left, ax_right
