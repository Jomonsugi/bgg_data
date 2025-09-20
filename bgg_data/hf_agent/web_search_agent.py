import argparse
import os
import requests
from io import BytesIO
from time import sleep
from urllib.parse import urljoin, urlparse

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, DuckDuckGoSearchTool, tool, InferenceClientModel
from smolagents.agents import ActionStep
from smolagents.cli import load_model


rulebook_search_request = """
You are a pdf rulebook search and downloadagent. You are searching for the rulebook for the board game {board_game}.
Once you find the rulebook in pdf format you will download it into a folder called hf_rulebooks
The most likely source of the rulebook will come from the board game's official website or on the board game's board game geek page under "offical links".  
Your first web search should always be "<board_game> official website". You then should scroll through the page.
DO NOT start with a search with the term "rulebook" or "rules". Just search for the board games "official website". This is usually effective.
The best way to verify if you have found the rulebook is to follow the url you have found and verify that the webpage looks like a rulebook.
You should only try another search if the official website does not have the rulebook.
The rulebook will be saved in the folder bgg_data/hf_agent/hf_rulebooks
If you cannot find the rulebook, you will return "No rulebook found".
"""


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a web browser automation script with a specified model.")
    parser.add_argument(
        "board_game",
        type=str,
        help="The board game name to search for (required)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=rulebook_search_request,
        help="Custom prompt to run with the agent (optional)",
    )
    # parser.add_argument(
    #     "--model-type",
    #     type=str,
    #     default="LiteLLMModel",
    #     help="The model type to use (e.g., OpenAIServerModel, LiteLLMModel, TransformersModel, InferenceClientModel)",
    # )
    # parser.add_argument(
    #     "--model-id",
    #     type=str,
    #     default="gpt-4o",
    #     help="The model ID to use for the specified model type",
    # )
    return parser.parse_args()


def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
    return


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()


@tool
def download_pdf(url: str, filename: str = None) -> str:
    """
    Downloads a PDF from the given URL and saves it to the hf_rulebooks folder.
    Args:
        url: The URL of the PDF to download
        filename: Optional custom filename (without .pdf extension). If not provided, will use the URL's filename or generate one.
    """
    try:
        # Create hf_rulebooks directory if it doesn't exist
        download_dir = "hf_rulebooks"
        os.makedirs(download_dir, exist_ok=True)
        
        # Set up session with retry adapter (pattern from codebase)
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/pdf, text/html;q=0.9,*/*;q=0.8'
        })
        
        # Install retry adapter (pattern from codebase)
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Handle special cases (patterns from codebase)
        original_url = url
        if 'dropbox.com' in url and ('?dl=0' in url or '&dl=0' in url):
            url = url.replace('?dl=0', '?dl=1').replace('&dl=0', '&dl=1')
            print(f"Converted Dropbox link to direct download: {url}")
        
        # Try Google Drive conversion
        try:
            if 'drive.google.com' in url and '/file/d/' in url:
                import re
                m = re.search(r"/file/d/([a-zA-Z0-9_-]+)/", url)
                if m:
                    file_id = m.group(1)
                    direct = f"https://drive.usercontent.google.com/uc?id={file_id}&export=download"
                    print(f"Converted Google Drive link to direct download: {direct}")
                    url = direct
        except Exception:
            pass
        
        # Generate filename if not provided
        if not filename:
            # Try to get filename from URL
            parsed_url = urlparse(url)
            url_filename = os.path.basename(parsed_url.path)
            if url_filename and url_filename.lower().endswith('.pdf'):
                filename = url_filename
            else:
                # Generate a filename based on current board game
                import time
                filename = f"rulebook_{int(time.time())}.pdf"
        
        # Ensure filename ends with .pdf
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        
        # Download with retries (pattern from codebase)
        success = False
        content = None
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                print(f"Download attempt {attempt + 1}/{max_retries}")
                
                response = session.get(url, timeout=30, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    print(f"Warning: Unexpected content type: {content_type}")
                
                # Download content
                content = response.content
                
                if content and len(content) > 0:
                    print(f"Download successful: {len(content)} bytes")
                    success = True
                    break
                else:
                    print("Download returned empty content")
                    content = None
                    
            except requests.exceptions.RequestException as e:
                print(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    print(f"All download attempts failed for {url}")
                    break
            except Exception as e:
                print(f"Unexpected error during download attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    break
        
        if not success or not content:
            # Try with referer and cookies from current browser context
            try:
                current_url = driver.current_url if driver else None
                referer = current_url if current_url else f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                extra_headers = {"Referer": referer, "Origin": referer}
                
                # Copy cookies from selenium if available
                if driver:
                    try:
                        cookies = driver.get_cookies() or []
                        for c in cookies:
                            try:
                                name = c.get('name')
                                value = c.get('value')
                                domain = c.get('domain') or None
                                path = c.get('path') or '/'
                                if name and value:
                                    session.cookies.set(name, value, domain=domain, path=path)
                            except Exception:
                                continue
                    except Exception:
                        pass
                
                print("Trying download with referer and cookies...")
                response = session.get(url, headers=extra_headers, timeout=30, stream=True)
                response.raise_for_status()
                content = response.content
                if content and len(content) > 0:
                    success = True
                    print(f"Download successful with referer/cookies: {len(content)} bytes")
            except Exception as e:
                print(f"Referer/cookies retry failed: {e}")
        
        if not success or not content:
            return f"Failed to download PDF from {original_url} after multiple attempts"
        
        # Validate content is actually a PDF
        def is_valid_pdf(content_bytes):
            try:
                return content_bytes.strip()[:4] == b'%PDF'
            except Exception:
                return False
        
        if not is_valid_pdf(content):
            # If not PDF, try to extract PDF link from HTML content
            try:
                if content.strip()[:15].lower().startswith(b'<!doctype html') or content.strip().lower().startswith(b'<html'):
                    print("Content appears to be HTML, attempting to extract PDF link...")
                    # Simple PDF link extraction
                    import re
                    pdf_links = re.findall(r'href=["\']([^"\']+\.pdf)["\']', content.decode('utf-8', errors='ignore'), re.IGNORECASE)
                    if pdf_links:
                        pdf_url = pdf_links[0]
                        if not pdf_url.startswith('http'):
                            pdf_url = urljoin(url, pdf_url)
                        print(f"Found PDF link in HTML: {pdf_url}")
                        # Try to download the PDF link
                        try:
                            response = session.get(pdf_url, timeout=30, stream=True)
                            response.raise_for_status()
                            content = response.content
                            if content and len(content) > 0 and is_valid_pdf(content):
                                print(f"Successfully downloaded PDF via HTML indirection: {len(content)} bytes")
                                success = True
                            else:
                                return f"PDF link found in HTML but download failed or invalid content"
                        except Exception as e:
                            return f"Failed to download PDF from extracted link: {e}"
                    else:
                        return f"Content is HTML but no PDF links found"
                else:
                    return f"Downloaded content is not a valid PDF (first 4 bytes: {content[:4]})"
            except Exception as e:
                return f"Error processing downloaded content: {e}"
        
        if not success:
            return f"Failed to download valid PDF content from {original_url}"
        
        # Save the PDF
        filepath = os.path.join(download_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(content)
        
        return f"Successfully downloaded PDF ({len(content)} bytes) to: {filepath}"
        
    except Exception as e:
        return f"Unexpected error: {str(e)}"
    finally:
        if 'session' in locals():
            session.close()


@tool
def get_current_url() -> str:
    """Returns the current URL of the browser."""
    return driver.current_url


def initialize_driver():
    """Initialize the Selenium WebDriver."""
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1000,1350")
    chrome_options.add_argument("--disable-pdf-viewer")
    chrome_options.add_argument("--window-position=0,0")
    return helium.start_chrome(headless=False, options=chrome_options)


def initialize_agent(model):
    """Initialize the CodeAgent with the specified model."""
    return CodeAgent(
        tools=[DuckDuckGoSearchTool(), go_back, close_popups, search_item_ctrl_f, download_pdf, get_current_url],
        model=model,
        additional_authorized_imports=["helium"],
        step_callbacks=[save_screenshot],
        max_steps=20,
        verbosity_level=2,
    )


helium_instructions = """
Use your web_search tool when you want to get Google search results.
Then you can use helium to access websites. Don't use helium for Google search, only for navigating websites!
Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>
You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>
If it's a link:
```py
click(Link("Top products"))
```<end_code>
If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.
To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>
When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
```py
close_popups()
```<end_code>
You can use .exists() to check for the existence of an element. For example:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
IMPORTANT: When you find a PDF rulebook URL, use the download_pdf tool to download it:
```py
download_pdf("https://example.com/rulebook.pdf", "game_name_rulebook")
```<end_code>
You can also get the current URL using the get_current_url tool to verify where you are before downloading.
Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>
If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
When you have modals or cookie banners on screen, you should get rid of them before you can click anything else.
"""


def main():
    # Load environment variables
    # For example to use an OpenAI model, create a local .env file with OPENAI_API_KEY="<your_open_ai_key_here>"
    load_dotenv() 

    # Parse command line arguments
    args = parse_arguments()

    # Initialize the model based on the provided arguments
    # model = load_model(args.model_type, args.model_id)
    model = InferenceClientModel(model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", provider="together")

    global driver
    driver = initialize_driver()
    agent = initialize_agent(model)

    # Build the prompt with the required board game
    prompt = args.prompt.format(board_game=args.board_game)

    # Run the agent with the provided prompt
    agent.python_executor("from helium import *")
    agent.run(prompt + helium_instructions)


if __name__ == "__main__":
    main()