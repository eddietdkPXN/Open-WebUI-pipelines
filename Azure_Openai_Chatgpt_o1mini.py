from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        # Configuration for Azure OpenAI
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_DEPLOYMENT_NAME: str
        AZURE_OPENAI_API_VERSION: str

    def __init__(self):
        # Set the name of the pipeline
        self.name = "Azure o1-mini"
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "https://xxx.openai.azure.com"), # replace here
                "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o1-mini"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            }
        )

    async def on_startup(self):
        # Function called when the server starts
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        # Function called when the server shuts down
        print(f"on_shutdown: {__name__}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Processes the input through the Azure OpenAI pipeline.

        Args:
            user_message (str): The message from the user.
            model_id (str): The ID of the model to use.
            messages (List[dict]): List of message dictionaries.
            body (dict): The request body containing parameters.

        Returns:
            Union[str, Generator, Iterator]: The response from Azure OpenAI.
        """
        print(f"pipe: {__name__}")

        print("Messages:", messages)
        print("User Message:", user_message)

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        url = (
            f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.valves.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions"
            f"?api-version={self.valves.AZURE_OPENAI_API_VERSION}"
        )

        allowed_params = {
            'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
            'enhancements', 'data_sources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
            'frequency_penalty', 'logit_bias', 'user', 'function_call', 'functions', 'tools',
            'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'
        }

        # Remap 'user' field if necessary
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"].get("id", str(body["user"]))

        # Filter the body to include only allowed parameters
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}

        # Log any dropped parameters
        if len(body) != len(filtered_body):
            dropped = set(body.keys()) - set(filtered_body.keys())
            print(f"Dropped params: {', '.join(dropped)}")

        try:
            response = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=filtered_body.get("stream", False),
            )

            response.raise_for_status()

            if filtered_body.get("stream"):
                return response.iter_lines()

            # Parse and return JSON response
            return response.json()

        except requests.RequestException as e:
            error_message = f"Request failed: {e}"
            if response is not None:
                try:
                    error_details = response.json()
                    error_message += f" | Details: {error_details}"
                except ValueError:
                    error_message += f" | Response Text: {response.text}"
            return f"Error: {error_message}"
