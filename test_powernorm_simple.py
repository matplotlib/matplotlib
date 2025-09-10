import sys
import os

# Add the lib directory to Python path
sys.path.insert(0, os.path.join(os.getcwd(), 'lib'))

# Import only the colors module directly
import importlib.util
spec = importlib.util.spec_from_file_location("colors", "lib/matplotlib/colors.py")
colors = importlib.util.module_from_spec(spec)

# Mock the dependencies that colors.py needs
import numpy as np
sys.modules['matplotlib._api'] = type(sys)('mock_api')
sys.modules['matplotlib._api'].check_getitem = lambda d, **kw: d.__getitem__
sys.modules['matplotlib'] = type(sys)('mock_matplotlib')
sys.modules['matplotlib.cbook'] = type(sys)('mock_cbook')
sys.modules['matplotlib.scale'] = type(sys)('mock_scale')
sys.modules['matplotlib._cm'] = type(sys)('mock_cm')
sys.modules['matplotlib.colorizer'] = type(sys)('mock_colorizer')

try:
    spec.loader.exec_module(colors)
    
    # Test PowerNorm with scale parameter
    norm = colors.PowerNorm(gamma=2.0, scale=2.0)
    print("SUCCESS! PowerNorm now accepts scale parameter")
    
    # Test that it works
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    result = norm(test_data)
    print(f"Input: {test_data}")
    print(f"Output: {result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
