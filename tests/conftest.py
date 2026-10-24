import os

# Tests never call a real LLM: they check the deterministic pipeline and the API contract.
os.environ["CX_LLM_PROVIDER"] = "mock"
