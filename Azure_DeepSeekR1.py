"""
description: A pipeline for generating text using the DeepSeek R1 model via Azure AI Inference API, with fixed stream handling.
requirements: azure-ai-inference
environment_variables: AZURE_INFERENCE_CREDENTIAL, AZURE_INFERENCE_ENDPOINT, MODEL_ID
"""

import os
import json
import logging
from typing import List, Union, Generator, Iterator, Tuple
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def pop_system_message(messages: List[dict]) -> Tuple[str, List[dict]]:
    """
    Extract the system message from the list of messages.
    
    Args:
    messages (List[dict]): List of message dictionaries.
    
    Returns:
    Tuple[str, List[dict]]: A tuple containing the system message (or empty string) and the updated list of messages.
    """
    system_message = ""
    updated_messages = []

    for message in messages:
        if message['role'] == 'system':
            system_message = message['content']
        else:
            updated_messages.append(message)

    return system_message, updated_messages


class Pipeline:

    class Valves(BaseModel):
        AZURE_INFERENCE_CREDENTIAL: str = ""
        AZURE_INFERENCE_ENDPOINT: str = ""
        MODEL_ID: str = "DeepSeekR1"

    def __init__(self):
        self.type = "manifold"
        self.id = "DeepSeekR1-azure"
        self.name = "DeepSeekR1-azure/"

        self.valves = self.Valves(
            **{
                "AZURE_INFERENCE_CREDENTIAL":
                os.getenv("AZURE_INFERENCE_CREDENTIAL",
                          "your-azure-inference-key-here"),
                "AZURE_INFERENCE_ENDPOINT":
                os.getenv("AZURE_INFERENCE_ENDPOINT",
                          "your-azure-inference-endpoint-here"),
                "MODEL_ID":
                os.getenv("MODEL_ID", "DeepSeekR1"),
            })
        self.update_client()

    def update_client(self):
        self.client = ChatCompletionsClient(
            endpoint=self.valves.AZURE_INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(
                self.valves.AZURE_INFERENCE_CREDENTIAL))

    def get_DeepSeekR1_models(self):
        return [
            {
                "id": "DeepSeekR1",
                "name": "DeepSeekR1"
            },
        ]

    async def on_startup(self):
        logger.info(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        logger.info(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_client()

    def pipelines(self) -> List[dict]:
        return self.get_DeepSeekR1_models()

    def pipe(self, user_message: str, model_id: str, messages: List[dict],
             body: dict) -> Union[str, Generator, Iterator]:
        try:
            logger.debug(
                f"Received request - user_message: {user_message}, model_id: {model_id}"
            )
            logger.debug(f"Messages: {json.dumps(messages, indent=2)}")
            logger.debug(f"Body: {json.dumps(body, indent=2)}")

            # Remove unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            system_message, messages = pop_system_message(messages)

            # Prepare messages for DeepSeekR1
            DeepSeekR1_messages = [SystemMessage(
                content=system_message)] if system_message else []
            DeepSeekR1_messages += [
                UserMessage(content=msg['content']) if msg['role'] == 'user'
                else SystemMessage(content=msg['content']) if msg['role']
                == 'system' else AssistantMessage(content=msg['content'])
                for msg in messages
            ]

            # Prepare the payload
            allowed_params = {
                'temperature', 'max_tokens', 'presence_penalty',
                'frequency_penalty', 'top_p'
            }
            filtered_body = {
                k: v
                for k, v in body.items() if k in allowed_params
            }

            logger.debug(f"Prepared DeepSeekR1 messages: {DeepSeekR1_messages}")
            logger.debug(f"Filtered body: {filtered_body}")

            is_stream = body.get("stream", False)
            if is_stream:
                return self.stream_response(DeepSeekR1_messages, filtered_body)
            else:
                return self.get_completion(DeepSeekR1_messages, filtered_body)
        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}", exc_info=True)
            return json.dumps({"error": str(e)})

    def stream_response(self, DeepSeekR1_messages: List[Union[SystemMessage, UserMessage, AssistantMessage]], params: dict) -> str:
        try:
            complete_response = ""
            response = self.client.complete(messages=DeepSeekR1_messages,
                                            model=self.valves.MODEL_ID,
                                            stream=True,
                                            **params)
            for update in response:
                if update.choices:
                    delta_content = update.choices[0].delta.content
                    if delta_content:
                        complete_response += delta_content
            return complete_response
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}", exc_info=True)
            return json.dumps({"error": str(e)})

    def get_completion(self, DeepSeekR1_messages: List[Union[SystemMessage, UserMessage, AssistantMessage]], params: dict) -> str:
        try:
            response = self.client.complete(messages=DeepSeekR1_messages,
                                            model=self.valves.MODEL_ID,
                                            **params)
            if response.choices:
                result = response.choices[0].message.content
                logger.debug(f"Completion result: {result}")
                return result
            else:
                logger.warning("No choices in completion response")
                return ""
        except Exception as e:
            logger.error(f"Error in get_completion: {str(e)}", exc_info=True)
            return json.dumps({"error": str(e)})


# TEST CASE TO RUN THE PIPELINE
if __name__ == "__main__":
    pipeline = Pipeline()

    messages = [{
        "role": "user",
        "content": "How many languages are in the world?"
    }]
    body = {
        "temperature": 0.5,
        "max_tokens": 150,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.8,
        "stream": True  # Change to True to test streaming
    }

    result = pipeline.pipe(user_message="How many languages are in the world?",
                           model_id="DeepSeekR1",
                           messages=messages,
                           body=body)

    # Handle streaming result
    if isinstance(result, str):
        content = json.dumps({"content": result}, ensure_ascii=False)
        print(content)
    else:
        complete_response = ""
        for part in result:
            content_delta = json.loads(part).get("delta")
            if content_delta:
                complete_response += content_delta

        print(json.dumps({"content": complete_response}, ensure_ascii=False))
