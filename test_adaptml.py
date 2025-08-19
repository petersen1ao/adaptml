#!/usr/bin/env python3
"""
Test script to verify AdaptML functionality
"""

import sys
sys.path.insert(0, '.')

# Test the basic functionality
from adaptml import quickstart, AdaptiveInference

print('=== AdaptML Quick Test ===')
print()

# Run the quickstart demo
print('Running quickstart demo...')
quickstart()

print()
print('=== Testing AdaptiveInference directly ===')

# Test direct usage
inference = AdaptiveInference()

# Test model registration
print('Testing model registration...')
model_id = inference.register_model(
    name='test_model',
    model_data={'type': 'mock', 'size': 'small'},
    metadata={'description': 'Test model for verification'}
)
print(f'Registered model with ID: {model_id}')

# Test inference
print('Testing inference...')
result = inference.predict(
    model_id=model_id,
    input_data='Hello AdaptML!',
    max_cost=0.01
)

print(f'Inference result: {result.prediction}')
print(f'Cost: ${result.cost:.6f}')
print(f'Latency: {result.latency:.3f}s')
print(f'Device: {result.device_used}')

print()
print('=== Contact Information ===')
import adaptml
print(f'Email: {adaptml.__email__}')
print(f'Website: {adaptml.__website__}')

print()
print('✅ AdaptML functionality test completed successfully!')
