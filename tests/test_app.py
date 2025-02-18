from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import pytest

def test_app_layout(dash_duo):
    from src.app import app

    # Start the Dash app in a test runner
    dash_duo.start_server(app)

    # Instead of looking for the Store component, wait for a visible element
    # For example, wait for the title or a control element that's definitely visible
    dash_duo.wait_for_element("h1", timeout=10)  # Wait for the main title
    
    # Or wait for multiple elements to ensure the layout is properly loaded
    dash_duo.wait_for_element("#permit-type", timeout=10)  # Wait for permit type dropdown
    
    # Then do your assertions
    assert "NYC Permits Dashboard" in dash_duo.find_element("h1").text
    assert "Controls" in dash_duo.find_element("body").text

# When initializing Chrome, use:
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))