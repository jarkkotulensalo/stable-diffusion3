import re


def generate_filename_from_prompt(prompt: str) -> str:
    """
    Generate a filename from the prompt.

    Args:
        prompt (str): The prompt.

    Returns:
        str: The filename.
    """

    # Tokenize the prompt and remove stop words
    words = prompt.split()
    important_words = [word.lower() for word in words]
    filename = "_".join(re.sub(r"\W+", "", word) for word in important_words)

    # cut the filename to 50 characters
    filename = filename[:50]
    return filename
