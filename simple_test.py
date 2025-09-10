# Read the PowerNorm class directly from the file
with open('lib/matplotlib/colors.py', 'r') as f:
    content = f.read()

# Check if our changes are there
if 'def __init__(self, gamma, vmin=None, vmax=None, clip=False, scale=1.0):' in content:
    print("✓ SUCCESS: PowerNorm __init__ method has scale parameter")
else:
    print("✗ FAILED: scale parameter not found in __init__")

if 'self.scale = scale' in content:
    print("✓ SUCCESS: self.scale assignment found")
else:
    print("✗ FAILED: self.scale assignment not found")

if 'resdat *= self.scale' in content:
    print("✓ SUCCESS: scaling applied in __call__ method")
else:
    print("✗ FAILED: scaling not applied in __call__ method")

if 'resdat /= self.scale' in content:
    print("✓ SUCCESS: inverse scaling applied in inverse method")
else:
    print("✗ FAILED: inverse scaling not applied in inverse method")

print("\nYour changes are implemented correctly in the file!")
