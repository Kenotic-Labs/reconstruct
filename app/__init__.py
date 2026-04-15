"""
Raya app package - sets critical environment paths BEFORE any model loads.
This file runs first when any app module is imported.
"""

import os

# Force all cache paths to S: drive (D: may not exist on this machine)
os.environ["HF_HOME"] = "D:/Nura/Env/hf_cache"
os.environ["HF_HUB_CACHE"] = "D:/Nura/Env/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "D:/Nura/Env/hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "D:/Nura/Env/hf_cache"
os.environ["TORCH_HOME"] = "D:/Nura/Env/torch_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
