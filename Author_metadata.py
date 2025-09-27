import os
from openai import OpenAI
import openai
import json
from pydantic import BaseModel
from typing import Optional
from Markdownparser import MDParser

class Author(BaseModel):
    """Represents a single author with their name, affiliation, and optional email."""
    Name: str
    Affiliation: str
    Email: str | None = None
    Author_Order: int | None = None

class AuthorList(BaseModel):
    """Represents a list of authors."""
    Authors: list[Author]

class AuthorMetadataExtractor:
    """
    A class to extract author metadata from text using an AI model.
    """
    def __init__(self, api_base_url: str, api_key: str, model: str, prompt_path = './prompts/author_extract_prompt.txt', llm_provider: str = "ollama"):
        """
        Initializes the AuthorMetadataExtractor.
        Args:
            api_base_url: The base URL for the OpenAI API.
            api_key: The API key for the OpenAI API.
            model: The name of the model to use.
            prompt_path: The path to the system prompt file.
            llm_provider: The LLM provider to use ("ollama" or "deepseek").
        """
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model = model
        self.llm_provider = llm_provider
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.system_prompt = f.read()
    
    def get_authors(self, content: str) -> AuthorList | str | None:
        """
        Gets structured author information from a block of text.
        Args:
            content: A string containing author names, affiliations, and emails.
        Returns:
            An AuthorList object, a refusal message, or None if an error occurs.
        """
        if self.llm_provider == "ollama":
            # Original code for ollama
            try:
                completion = self.client.beta.chat.completions.parse(
                    temperature=0,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content}
                    ],
                    response_format=AuthorList,
                )
               
                pet_response = completion.choices[0].message
                if pet_response.parsed:
                    return pet_response.parsed
                elif pet_response.refusal:
                    return pet_response.refusal
            except openai.LengthFinishReasonError as e:
                return f"Too many tokens: {e}"
            except Exception as e:
                return f"An error occurred: {e}"
            return None
            
        elif self.llm_provider == "deepseek":
            # Use function calling for deepseek
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "extract_authors",
                        "description": "Extract author information from the given text.",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "authors": {
                                    "type": "array",
                                    "description": "A list of authors with their information",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {
                                                "type": "string",
                                                "description": "The full name of the author"
                                            },
                                            "affiliation": {
                                                "type": "string",
                                                "description": "The institutional affiliation of the author"
                                            },
                                            "email": {
                                                "type": ["string", "null"],
                                                "description": "The email address of the author if available"
                                            },
                                            "author_order": {
                                                "type": ["integer", "null"],
                                                "description": "The order of the author in the author list"
                                            }
                                        },
                                        "required": ["name", "affiliation"]
                                    }
                                }
                            },
                            "required": ["authors"]
                        }
                    }
                }
            ]
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content}
                    ],
                    tools=tools, # type: ignore
                    tool_choice={"type": "function", "function": {"name": "extract_authors"}}
                )
                
                # Parse the function call response
                choice = response.choices[0]
                message = choice.message
                
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    arguments = tool_call.function.arguments # type: ignore
                    if isinstance(arguments, str):
                        data = json.loads(arguments)
                    else:
                        data = arguments
                    
                    # Convert to AuthorList
                    authors = []
                    for author_data in data.get('authors', []):
                        author = Author(
                            Name=author_data.get('name', ''),
                            Affiliation=author_data.get('affiliation', ''),
                            Email=author_data.get('email'),
                            Author_Order=author_data.get('author_order')
                        )
                        authors.append(author)
                    
                    return AuthorList(Authors=authors)
                    
            except openai.LengthFinishReasonError as e:
                return f"Too many tokens: {e}"
            except Exception as e:
                return f"An error occurred: {e}"
            return None