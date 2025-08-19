# Fix the quickstart function to properly share the inference instance

import re

# Read the file
with open('adaptml/core.py', 'r') as f:
    content = f.read()

# Replace the create_demo_models function to accept an inference instance
new_create_demo_models = '''def create_demo_models(inference: 'AdaptiveInference' = None) -> Dict[str, str]:
    """Create demo models for testing"""
    if inference is None:
        inference = AdaptiveInference()
    models = {}
    
    # Register mock models
    models["small"] = inference.register_model(
        "small_classifier",
        {"type": "mock", "size": "small"},
        {"description": "Small mock classifier"}
    )
    
    models["medium"] = inference.register_model(
        "medium_transformer",
        {"type": "mock", "size": "medium"},
        {"description": "Medium transformer model"}
    )
    
    return models'''

# Replace the old function
pattern = r'def create_demo_models\(\) -> Dict\[str, str\]:.*?return models'
content = re.sub(pattern, new_create_demo_models, content, flags=re.DOTALL)

# Also fix the quickstart function to pass the inference instance
content = content.replace(
    'models = create_demo_models()',
    'models = create_demo_models(inference)'
)

# Write back
with open('adaptml/core.py', 'w') as f:
    f.write(content)

print("Fixed create_demo_models and quickstart functions")
