
import re

def transform_dict(input_dict):
    function_dict = input_dict['function']
    description = function_dict['description']
    name = function_dict['name']
    parameters = function_dict['parameters']

    # Transform the function name
    name = re.sub(r'[-_]', '_', name)

    # Transform the parameter types
    for prop, value in parameters['properties'].items():
        if value['type'] == 'number':
            value['type'] = 'integer'

    # Generate the function description
    param_descriptions = []
    for prop, value in parameters['properties'].items():
        param_desc = f"{prop} ({value['type']}): {value.get('description', 'No description')}"
        param_descriptions.append(param_desc)

    param_str = '\n'.join([f"        {desc}" for desc in param_descriptions])
    func_description = f"{name}({', '.join([f'{prop}: {value['type']}' for prop, value in parameters['properties'].items()])}) -> dict | str - {description}\n    \n    Args:\n{param_str}"

    # Remove the 'description' key from parameters['properties']
    for prop in parameters['properties']:
        parameters['properties'][prop].pop('description', None)
        
    # Update the function dictionary
    function_dict['description'] = func_description
    function_dict['name'] = name
    function_dict['parameters'] = {
        'properties': parameters['properties'],
        'required': parameters['required'],
        'type': parameters['type']
    }

    return input_dict

# Example usage
input_dict = {
    'function': {
        'description': 'Swap token of user',
        'name': 'swap-token',
        'parameters': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'additionalProperties': False,
            'properties': {
                'input_amount': {'description': 'The input amount to convert', 'type': 'number'},
                'input_symbol': {'description': 'The input symbol to convert', 'type': 'string'},
                'output_symbol': {'description': 'The output symbol to convert', 'type': 'string'}
            },
            'required': ['input_symbol', 'input_amount', 'output_symbol'],
            'type': 'object'
        }
    }
}

output_dict = transform_dict(input_dict)
from pprint import pprint
pprint(output_dict)