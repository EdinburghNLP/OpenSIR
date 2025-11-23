import re
import textwrap
import logging
from typing import Literal, Optional

logger = logging.getLogger(__name__)


def clean_code_block(
    text: str,
    match_occurrence: Literal["first", "last"] = "first",
    dedent: bool = True,
) -> str:
    """Extracts code from within markdown code blocks.

    This function uses a regular expression to find all markdown code blocks
    (e.g., ```python\\ncode\\n``` or ```\\ncode\\n```) and extracts the code content.
    It can be configured to return either the first or the last code block found.
    It also handles indented code blocks, where the entire block including the
    triple backticks is indented.

    Args:
        text (str): The text containing markdown code blocks.
        match_occurrence (Literal["first", "last"]): Specifies which code block to extract.
            Defaults to "first".
            - "first": Extracts the content of the first code block found.
            - "last": Extracts the content of the last code block found.
        dedent (bool): Whether to remove common leading whitespace from all lines
            of the extracted code. Defaults to True.

    Returns:
        str: The extracted code content. If no valid code block is found,
             returns the original text stripped of leading/trailing whitespace.
    """
    # Regex to find code blocks: (indentation)```optional_lang\\n(content)\\n(same indentation)```
    # - ([ \t]*)?: Captures any leading whitespace (indentation) before the opening triple backticks.
    #   This is group 1, which we'll use to match the same indentation before the closing backticks.
    # - ```: Matches the opening triple backticks.
    # - (?:[a-zA-Z0-9_+\-\.#\\s]*)?: Optionally matches a language specifier.
    #   This is a non-capturing group. Allows letters, numbers, underscore,
    #   plus, hyphen, dot, hash, and whitespace for language names like 'C++' or 'Objective-C#'.
    # - [ \t]*: Optionally matches any spaces or tabs after the language specifier.
    # - \\n: Matches the newline after the opening marker line.
    # - (.*?): Captures the actual code content. This is group 2.
    #   - .*?: Matches any character (including newlines with re.DOTALL) in a non-greedy way.
    # - \\n: Matches the newline before the closing marker.
    # - \\1: Backreference to match the same indentation as captured in group 1.
    # - ```: Matches the closing triple backticks.
    # re.DOTALL: Makes '.' match any character, including newlines. This is crucial
    #            for multi-line code blocks.
    pattern = re.compile(
        r"([ \t]*)```(?:[a-zA-Z0-9_+\-\.#\\s]*)?[ \t]*\n(.*?)\n\1```",
        re.DOTALL,
    )

    matches = pattern.findall(text)

    if not matches:
        return (
            text.strip()
        )  # For no match, strip the original text as a fallback

    # Extract the code content (group 2)
    code_blocks = [match[1] for match in matches]

    if not code_blocks:
        return text.strip()

    selected_code = (
        code_blocks[0] if match_occurrence == "first" else code_blocks[-1]
    )

    # Remove common leading whitespace if requested
    if dedent:
        return textwrap.dedent(selected_code)

    return selected_code


def insert_before_theorem(text: str, new_line: str) -> str:
    """Inserts a new line before any line that starts with 'theorem'.

    This function searches for lines that start with 'theorem' (case-sensitive)
    and inserts the provided new line text immediately before each such line.
    The function preserves all whitespace and formatting of the original text.

    Args:
        text (str): The original text to modify.
        new_line (str): The line to insert before 'theorem' lines.

    Returns:
        str: The modified text with the new line inserted before each 'theorem' line.
            If no 'theorem' line is found, returns the original text unchanged.
    """
    # Use regex to find lines that start with 'theorem'
    # The pattern looks for:
    # - Start of line (^) or a newline character (\n)
    # - Followed by 'theorem' (case-sensitive)
    # - Followed by a space or other whitespace character
    pattern = re.compile(r"(^|\n)(theorem\s)", re.MULTILINE)

    # Replace with the original match plus the new line
    # Group 1 captures the newline or start of string
    # Group 2 captures 'theorem ' part
    # We insert the new line between these groups
    return pattern.sub(r"\1" + new_line + r"\n\2", text)


def sanitize_text(text: str) -> str:
    """
    Sanitizes a string to handle various Unicode encoding issues robustly.
    
    This function attempts multiple strategies to clean text that may contain
    invalid UTF-8 sequences, surrogate pairs, or other encoding problems.
    It uses a fallback approach, trying progressively more aggressive
    sanitization methods until the text can be safely processed.
    
    Args:
        text (str): The text to sanitize. If not a string, returns unchanged.
        
    Returns:
        str: The sanitized text that is safe for UTF-8 processing.
        
    Raises:
        None: This function is designed to never raise exceptions, always
              returning some form of usable text.
    """
    # Handle None input specifically
    if text is None:
        logger.debug("sanitize_text received None input")
        return None
    
    # Handle non-string inputs
    if not isinstance(text, str):
        logger.debug(f"sanitize_text received non-string input: {type(text)}")
        return text
    
    # Handle empty strings
    if not text:
        return text
    
    # Strategy 1: Try normal UTF-8 encoding/decoding (most common case)
    try:
        # Test if the string is already valid UTF-8
        text.encode('utf-8').decode('utf-8')
        return text
    except UnicodeError:
        logger.debug("sanitize_text: Strategy 1 (normal UTF-8) failed, trying fallbacks")
    
    # Strategy 2: Use 'replace' error handler to substitute invalid sequences
    try:
        sanitized = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        logger.info("sanitize_text: Used 'replace' error handler for text sanitization")
        return sanitized
    except UnicodeError:
        logger.debug("sanitize_text: Strategy 2 ('replace' handler) failed")
    
    # Strategy 3: Use 'ignore' to skip problematic bytes
    try:
        sanitized = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        logger.warning("sanitize_text: Used 'ignore' error handler - some characters may be lost")
        return sanitized
    except UnicodeError:
        logger.debug("sanitize_text: Strategy 3 ('ignore' handler) failed")
    
    # Strategy 4: Use 'backslashreplace' for debugging problematic sequences
    try:
        sanitized = text.encode('utf-8', errors='backslashreplace').decode('utf-8', errors='replace')
        logger.warning("sanitize_text: Used 'backslashreplace' - text contains escape sequences")
        return sanitized
    except UnicodeError:
        logger.error("sanitize_text: All strategies failed, using repr() as last resort")
    
    # Final fallback: Convert to repr() to make it safe (should never reach here)
    try:
        return repr(text)[1:-1]  # Remove the surrounding quotes from repr()
    except Exception as e:
        logger.error(f"sanitize_text: Final fallback failed: {e}")
        return "<SANITIZATION_FAILED>"
