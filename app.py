#!/usr/bin/env python

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
# import wave
# import contextlib
# import base64
from synthtts import TtsService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)






coqui = TtsService()


class Prompt(BaseModel):
    """
    Class that represents the model for a text prompt. 
    It inherits from Pydantic's BaseModel and includes a single attribute, prompt.
    """
    prompt: str


@app.post('/v1/tts')
async def get_prompt_data(prompt_data: Prompt):
    """
    FastAPI route that takes in a text prompt, generates a synthesized
    speech audio file using the TtsService, and returns the response.

    Parameters:
    - prompt_data : Instance of the Prompt model, expecting a JSON body
                    with a "prompt" field in the POST request.

    Returns:
    - dict: If prompt text was received in request, it returns a dictionary with the
            details of the synthesized speech, including word boundaries and audio file details.
    - dict: If no prompt text was received, it returns a dictionary with a detail message.
    """

    if prompt_data.prompt:
        search_query = prompt_data.prompt

        response = coqui.generate_from_text(search_query, "./")

        for key, value in response.items():
            print(f"{key}: {value}")

        return response

    else:
        return {"detail": "Prompt parameter is missing"}
