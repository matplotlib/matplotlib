from matplotlib.config.parse_user_config import (
        update_config_from_dict_path, update_config_dict_from_user_config
)


def test_set_config_dict_path():
    config_dict = {}
    update_config_from_dict_path(config_dict, 'a:b:c', 1)
    assert config_dict['a']['b']['c'] == 1


def test_set_config_dict_values_user_config():
    user_config = {'lines.linewidth': 100}
    config_dict = {}
    update_config_dict_from_user_config(config_dict, user_config)

    value = config_dict['collections.LineCollection']['__init__']['linewidths']
    assert value == 100
    value = config_dict['contour.ContourSet']['__init__']['linewidths']
    assert value == 100
    value = config_dict['lines.Line2D']['__init__']['linewidth']
    assert value == 100


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
