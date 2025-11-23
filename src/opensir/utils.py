import logging
import openai
import time

logger = logging.getLogger(__name__)


def openai_request_with_retry(api_call_func, max_retries=50, delay_seconds=10):
    """
    Wrapper function to retry OpenAI API calls on connection exceptions.

    Args:
        api_call_func: A callable that makes the OpenAI API request
        max_retries: Maximum number of retry attempts (default: 50)
        delay_seconds: Delay between retries in seconds (default: 10)

    Returns:
        The result of the API call

    Raises:
        The last exception encountered if all retries are exhausted
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            return api_call_func()
        except openai.APIConnectionError as e:
            if attempt < max_retries:
                logger.warning(
                    "OpenAI connection error (attempt"
                    f" {attempt + 1}/{max_retries + 1}): {e}. Retrying in"
                    f" {delay_seconds} seconds..."
                )
                time.sleep(delay_seconds)
            else:
                logger.error(
                    "OpenAI connection failed after"
                    f" {max_retries + 1} attempts. Last error: {e}"
                )
                raise
        except Exception as e:
            # For non-connection errors, don't retry
            logger.error(f"OpenAI API error (non-connection): {e}")
            raise
