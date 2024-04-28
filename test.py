example = "USER: swap A to 100 B\nASSISTANT:\n<tool_call>\n{'arguments': {'input_symbol': 'A', 'input_amount': null, 'output_symbol': 'B', 'output_amount': 100}, 'name': 'swap_token'}\n</tool_call>"

origin_examples = [
    {
        "example": "```\nSYSTEM: You are a helpful assistant who has access to functions. Use them if required\n<tools>[\n {\n \"name\": \"calculate_distance\",\n \"description\": \"Calculate the distance between two locations\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"origin\": {\n \"type\": \"string\",\n \"description\": \"The starting location\"\n },\n \"destination\": {\n \"type\": \"string\",\n \"description\": \"The destination location\"\n },\n \"mode\": {\n \"type\": \"string\",\n \"description\": \"The mode of transportation\"\n }\n },\n \"required\": [\n \"origin\",\n \"destination\",\n \"mode\"\n ]\n }\n },\n {\n \"name\": \"generate_password\",\n \"description\": \"Generate a random password\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"length\": {\n \"type\": \"integer\",\n \"description\": \"The length of the password\"\n }\n },\n \"required\": [\n \"length\"\n ]\n }\n }\n]\n\n</tools>\nUSER: Hi, I need to know the distance from New York to Los Angeles by car.\nASSISTANT:\n<tool_call>\n{\"arguments\": {\"origin\": \"New York\",\n \"destination\": \"Los Angeles\", \"mode\": \"car\"}, \"name\": \"calculate_distance\"}\n</tool_call>\n```\n"
    },
    {
        "example": "```\nSYSTEM: You are a helpful assistant with access to functions. Use them if required\n<tools>[\n {\n \"name\": \"calculate_distance\",\n \"description\": \"Calculate the distance between two locations\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"origin\": {\n \"type\": \"string\",\n \"description\": \"The starting location\"\n },\n \"destination\": {\n \"type\": \"string\",\n \"description\": \"The destination location\"\n },\n \"mode\": {\n \"type\": \"string\",\n \"description\": \"The mode of transportation\"\n }\n },\n \"required\": [\n \"origin\",\n \"destination\",\n \"mode\"\n ]\n }\n },\n {\n \"name\": \"generate_password\",\n \"description\": \"Generate a random password\",\n \"parameters\": {\n \"type\": \"object\",\n \"properties\": {\n \"length\": {\n \"type\": \"integer\",\n \"description\": \"The length of the password\"\n }\n },\n \"required\": [\n \"length\"\n ]\n }\n }\n]\n\n</tools>\nUSER: Can you help me generate a random password with a length of 8 characters?\nASSISTANT:\n<tool_call>\n{\"arguments\": {\"length\": 8}, \"name\": \"generate_password\"}\n</tool_call>\n```"
    }
]

print(example)