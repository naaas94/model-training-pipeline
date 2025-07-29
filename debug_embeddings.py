import pandas as pd
import json
import ast

# Load the data
df = pd.read_csv('src/data/balanced_dataset_20250728.csv')
emb_str = df['embeddings'].iloc[0]

print('Original string length:', len(emb_str))
print('First 100 chars:', repr(emb_str[:100]))
print('Last 100 chars:', repr(emb_str[-100:]))

# Try JSON parsing
try:
    parsed_json = json.loads(emb_str)
    print('JSON parse successful, length:', len(parsed_json))
    print('First 5 values:', parsed_json[:5])
except Exception as e:
    print('JSON parse failed:', e)

# Try AST parsing
try:
    parsed_ast = ast.literal_eval(emb_str)
    print('AST parse successful, length:', len(parsed_ast))
    print('First 5 values:', parsed_ast[:5])
except Exception as e:
    print('AST parse failed:', e) 