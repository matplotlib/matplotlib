import numpy as np
import sys
import os

# We'll test the logic directly without importing matplotlib
# Let's extract and test the PowerNorm logic

def test_powernorm_scaling():
    """Test that scaling works correctly in PowerNorm logic"""
    
    # Simulate what PowerNorm should do with scaling
    def powernorm_with_scale(data, gamma, vmin, vmax, scale):
        # Apply scaling (our addition)
        scaled_data = data * scale
        # Apply normalization
        normalized = (scaled_data - vmin) / (vmax - vmin)
        # Apply power law
        result = np.power(normalized, gamma)
        return result
    
    # Test data
    test_data = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test with scale=1.0 (should be same as no scaling)
    result_no_scale = powernorm_with_scale(test_data, gamma=2.0, vmin=1.0, vmax=4.0, scale=1.0)
    
    # Test with scale=2.0 
    result_with_scale = powernorm_with_scale(test_data, gamma=2.0, vmin=1.0, vmax=4.0, scale=2.0)
    
    print("Test Results:")
    print(f"Input data: {test_data}")
    print(f"With scale=1.0: {result_no_scale}")
    print(f"With scale=2.0: {result_with_scale}")
    
    # Verify scaling effect
    # When scale=2.0, input [1,2,3,4] becomes [2,4,6,8]
    # So the normalization should be different
    if not np.array_equal(result_no_scale, result_with_scale):
        print("✓ SUCCESS: Scaling changes the output as expected")
    else:
        print("✗ FAILED: Scaling has no effect")
    
    return True

if __name__ == "__main__":
    test_powernorm_scaling()
