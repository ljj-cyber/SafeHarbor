import os


def model_config(model):
    """Build a single model config entry for AutoGen.

    API key and base URL are read from environment variables so the file
    contains no hard-coded secrets:

        - ``GUARDAGENT_API_KEY`` (fallback: ``OPENAI_API_KEY``)
        - ``GUARDAGENT_API_BASE`` (fallback: ``OPENAI_BASE_URL``)
    """
    api_key = os.getenv("GUARDAGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("GUARDAGENT_API_BASE") or os.getenv("OPENAI_BASE_URL", "")

    config = {
        "model": "gpt-3.5-turbo" if model == "gpt-3.5-turbo" else "gpt-4o",
        "api_key": api_key,
    }
    if base_url:
        config["base_url"] = base_url
    return config


def llm_config_list(seed, config_list):
    return {
        "functions": [
            {
                "name": "python",
                "description": "run the entire code and return the execution result. Only generate the code.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cell": {
                            "type": "string",
                            "description": "Valid Python code to execute.",
                        }
                    },
                    "required": ["cell"],
                },
            },
        ],
        "config_list": config_list,
        "timeout": 120,
        "cache_seed": seed,
        "temperature": 0,
    }
