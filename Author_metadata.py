from openai import OpenAI
import openai
from pydantic import BaseModel
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
    def __init__(self, api_base_url: str, api_key: str, model: str, prompt_path = './prompts/author_extract_prompt.txt'):
        """
        Initializes the AuthorMetadataExtractor.

        Args:
            api_base_url: The base URL for the OpenAI API.
            api_key: The API key for the OpenAI API.
            model: The name of the model to use.
            prompt_path: The path to the system prompt file.
        """
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model = model
        with open(prompt_path, 'r') as f:
            self.system_prompt = f.read()

    def get_authors(self, content: str) -> AuthorList | str | None:
        """
        Gets structured author information from a block of text.

        Args:
            content: A string containing author names, affiliations, and emails.

        Returns:
            An AuthorList object, a refusal message, or None if an error occurs.
        """
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
