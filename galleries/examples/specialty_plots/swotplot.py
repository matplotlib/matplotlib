"""
============
SWOT Diagram
============

SWOT analysis (or SWOT matrix) is a strategic planning and strategic management
technique used to help a person or organization identify Strengths, Weaknesses,
Opportunities, and Threats related to business competition or project planning.
It is sometimes called situational assessment or situational analysis.
Additional acronyms using the same components include TOWS and WOTS-UP.
Source: https://en.wikipedia.org/wiki/SWOT_analysis

"""
from typing import Tuple

import matplotlib.pyplot as plt


def draw_quadrant(q_data: dict, ax: plt.Axes, q_color: str,
                  q_x: Tuple[float, float, float, float],
                  q_y: Tuple[float, float, float, float],
                  q_title_pos: Tuple[float, float] = (0.0, 0.0),
                  q_title_props: dict = dict(boxstyle='round',
                                             facecolor="white",
                                             alpha=1.0),
                  q_point_props: dict = dict(boxstyle='round',
                                             facecolor="white",
                                             alpha=1.0)
                  ) -> None:
    """
    Draw a quadrant of the SWOT plot

    Parameters
    ----------
    q_data : dict
        Data structure passed to populate the quadrant.
        Format 'key':Tuple(float,float)
    ax : plt.Axes
        Matplotlib current axes
    q_color : str
        Quadrant color.
    q_x : Tuple[float, float, float, float]
        Quadrant X coordinates.
    q_y : Tuple[float, float, float, float]
        Quadrant Y coordinates.
    q_title_pos : Tuple[float,float]
        Plot title relative position
    q_title_props : dict, optional
        Title box style properties. The default is dict(boxstyle='round',
        facecolor="white", alpha=1.0).
    q_point_props : dict, optional
        Point box style properties. The default is dict(boxstyle='round',
        facecolor="white", alpha=1.0).

    Returns
    -------
    None

    """
    # draw filled rectangle
    ax.fill(q_x, q_y, color=q_color)

    # draw quadrant title
    ax.text(x=q_title_pos[0], y=q_title_pos[1],
            s=str.upper(list(q_data.keys())[0]), fontsize=12, weight='bold',
            verticalalignment='center', horizontalalignment='center',
            bbox=q_title_props)

    # fill quadrant with points
    for k, v in q_data[list(q_data.keys())[0]].items():
        ax.text(x=v[0], y=v[1], s=k, fontsize=12, weight='bold',
                verticalalignment='center', horizontalalignment='center',
                bbox=q_point_props)

    return None


def swotplot(p_data: dict, ax: plt.Axes,
             s_color: str = "forestgreen", w_color: str = "gold",
             o_color: str = "skyblue", t_color: str = "firebrick",
             left_margin: float = 0.05, right_margin: float = 0.05,
             top_margin: float = 0.05, bottom_margin: float = 0.05,
             ) -> None:
    """
    Draw SWOT plot

    Parameters
    ----------
    p_data : dict
        Data structure passed to populate the plot.
    ax : matplotlib.pyplot.Axes
        axes in which to draw the plot
    s_color : float, optional
        Strength quadrant color.
    w_color : float, optional
        Weakness quadrant color.
    o_color : float, optional
        Opportunity quadrant color.
    t_color : float, optional
        Threat quadrant color.
    left_margin : float, optional
        Left spacing from frame border. The default is 0.05.
    right_margin : float, optional
        Right spacing from frame border. The default is 0.05.
    top_margin : float, optional
        Top spacing from frame border. The default is 0.05.
    bottom_margin : float, optional
        Bottom spacing from frame border. The default is 0.05.

    Returns
    -------
    None

    """
    # format axis
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    # draw s quadrant
    draw_quadrant(q_data=dict(filter(lambda i: i[0] in list(p_data.keys())[0],
                                     p_data.items())),
                  ax=ax,
                  q_color=s_color, q_x=(left_margin, left_margin, 0.5, 0.5),
                  q_y=(0.5, 1 - top_margin, 1 - top_margin, 0.5),
                  q_title_pos=(0.25 + left_margin / 2, 1 - top_margin))

    # draw w quadrant
    draw_quadrant(q_data=dict(filter(lambda i: i[0] in list(p_data.keys())[1],
                                     p_data.items())),
                  ax=ax,
                  q_color=w_color, q_x=(0.5, 0.5, 1 - right_margin, 1 - right_margin),
                  q_y=(0.5, 1 - top_margin, 1 - top_margin, 0.5),
                  q_title_pos=(0.25 + (1 - right_margin) / 2, 1 - top_margin))

    # draw o quadrant
    draw_quadrant(q_data=dict(filter(lambda i: i[0] in list(p_data.keys())[2],
                                     p_data.items())),
                  ax=ax,
                  q_color=o_color, q_x=(left_margin, left_margin, 0.5, 0.5),
                  q_y=(bottom_margin, 0.5, 0.5, bottom_margin),
                  q_title_pos=(0.25 + left_margin / 2, bottom_margin))

    # draw t quadrant
    draw_quadrant(q_data=dict(filter(lambda i: i[0] in list(p_data.keys())[3],
                                     p_data.items())),
                  ax=ax,
                  q_color=t_color, q_x=(0.5, 0.5, 1 - right_margin, 1 - right_margin),
                  q_y=(bottom_margin, 0.5, 0.5, bottom_margin),
                  q_title_pos=(0.25 + (1 - right_margin) / 2, bottom_margin))

    return None


# USER DATA
# relative x_pos,y_pos make it easy to manage particular cases such as
# soft categorization
data = {'strength': {'SW automation': (0.3, 0.6),
                     'teamwork': (0.4, 0.5)
                     },
        'weakness': {'delayed payment': (0.5, 0.7),
                     'hard to learn': (0.7, 0.9)
                     },
        'opportunity': {'dedicated training': (0.3, 0.3),
                        'just-in-time development': (0.2, 0.1)
                        },
        'threat': {'project de-prioritization': (0.7, 0.3),
                   'external consultancy': (0.5, 0.1)
                   },
        }

# SWOT plot generation
fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')

swotplot(data, ax)
