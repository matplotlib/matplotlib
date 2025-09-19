import numpy as np
import sys
import os

# Add the lib directory to Python path so we can import matplotlib
sys.path.insert(0, 'lib')

try:
    import matplotlib.colors as mcolors
    print("✓ Successfully imported matplotlib.colors")
    
    def test_PowerNorm_scale_complete():
        """Complete test for PowerNorm scale parameter"""
        print("\n=== Testing PowerNorm Scale Parameter ===")
        
        # Test basic functionality with scale parameter
        a = np.array([1, 2, 3, 4], dtype=float)
        print(f"Test data: {a}")
        
        # Test with scale=1.0 (should be same as no scaling)
        pnorm_no_scale = mcolors.PowerNorm(gamma=2, vmin=1, vmax=4, scale=1.0)
        pnorm_default = mcolors.PowerNorm(gamma=2, vmin=1, vmax=4)
        
        result_no_scale = pnorm_no_scale(a)
        result_default = pnorm_default(a)
        
        print(f"With scale=1.0: {result_no_scale}")
        print(f"Default (no scale): {result_default}")
        
        # Results should be identical when scale=1.0
        if np.allclose(result_no_scale, result_default):
            print("✓ SUCCESS: scale=1.0 produces same results as default")
        else:
            print("✗ FAILED: scale=1.0 should produce same results as default")
            return False
        
        # Test with scale=2.0 (should produce different results)
        pnorm_scaled = mcolors.PowerNorm(gamma=2, vmin=1, vmax=4, scale=2.0)
        result_scaled = pnorm_scaled(a)
        
        print(f"With scale=2.0: {result_scaled}")
        
        # Results should be different when scaling is applied
        if not np.allclose(result_scaled, result_no_scale):
            print("✓ SUCCESS: scale=2.0 produces different results")
        else:
            print("✗ FAILED: scale=2.0 should produce different results")
            return False
        
        # Test inverse function works correctly with scaling
        a_roundtrip = pnorm_scaled.inverse(result_scaled)
        print(f"Roundtrip test: {a} -> {result_scaled} -> {a_roundtrip}")
        
        if np.allclose(a, a_roundtrip):
            print("✓ SUCCESS: inverse function works with scaling")
        else:
            print("✗ FAILED: inverse function doesn't work correctly")
            print(f"Expected: {a}")
            print(f"Got: {a_roundtrip}")
            return False
        
        # Test manual calculation
        expected_scaled_data = a * 2.0  # [2, 4, 6, 8]
        manual_norm = (expected_scaled_data - 1) / 3  # normalize with vmin=1, vmax=4
        manual_power = np.power(manual_norm, 2)  # Apply gamma=2
        
        print(f"Manual calculation: {manual_power}")
        print(f"PowerNorm result: {result_scaled}")
        
        if np.allclose(result_scaled, manual_power):
            print("✓ SUCCESS: manual calculation matches PowerNorm result")
        else:
            print("✗ FAILED: manual calculation doesn't match")
            return False
        
        print("\n=== All tests passed! ===")
        return True
    
    # Run the test
    test_PowerNorm_scale_complete()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("You need to rebuild matplotlib. Try: pip install -e . --no-build-isolation")
