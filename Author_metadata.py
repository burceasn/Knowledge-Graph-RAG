from openai import OpenAI
import openai
from pydantic import BaseModel
from Markdownparser import MDParser

class Author(BaseModel):
    """Represents a single author with their name, affiliation, and optional email."""
    Name: str
    Affiliation: str
    Email: str | None = None

class AuthorList(BaseModel):
    """Represents a list of authors."""
    Authors: list[Author]

class AuthorMetadataExtractor:
    """
    A class to extract author metadata from text using an AI model.
    """
    def __init__(self, api_base_url: str, api_key: str, model: str, prompt_path: str):
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

if __name__ == '__main__':
    # Configuration
    API_BASE_URL = "http://localhost:11434/v1"
    API_KEY = "ollama"
    MODEL = "qwen3:4b"
    PROMPT_PATH = './prompts/author_extract_prompt.txt'
    
    # Example Usage
    with open('example.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Use MDParser to get the content between the title and the abstract
    parser = MDParser(markdown_content)
    
    # The content we want is between the H1 title and the H6 abstract.
    # A simple way is to get all content under the H1 and then split it.
    title_h1 = parser.get_heading("", level=1)
    content_under_h1 = ""
    if title_h1:
        content_under_h1 = parser.get_content(title_h1, level=1)

    author_text = None
    if content_under_h1:
        # The parser gets all content until the next H1. We only want the part before the abstract.
        # We can find the abstract heading and take the text before it.
        if 'Abstract' in content_under_h1:
            author_text = content_under_h1.split('Abstract')[0].strip()
        else:
            author_text = content_under_h1.strip()

    # Create an instance of the extractor
    extractor = AuthorMetadataExtractor(
        api_base_url=API_BASE_URL,
        api_key=API_KEY,
        model=MODEL,
        prompt_path=PROMPT_PATH
    )

    if author_text:
        print("Extracted Author Text (using MDParser):")
        print(author_text)
        print("-" * 20)
        
        # Get the structured author information
        authors_result = extractor.get_authors(author_text)
        
        if isinstance(authors_result, AuthorList):
            print("Successfully parsed authors:")
            print(authors_result)
        elif isinstance(authors_result, str):
            print(f"Model returned a message: {authors_result}")
        else:
            print("Failed to extract authors.")
    else:
        print("Could not find author metadata using MDParser.")
