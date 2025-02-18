import pytest
from src.app_instance import app
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
import dash.testing.browser

@pytest.fixture
def dash_app():
    # Ensure that the app doesn't complain about missing callbacks
    app.config.suppress_callback_exceptions = True
    return app

@pytest.hookimpl
def pytest_setup_options():
    from selenium.webdriver.chrome.options import Options
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    # Use a temporary directory for user data
    options.add_argument('--user-data-dir=/tmp/chrome-data')
    return options

def pytest_configure():
    ChromeDriverManager().install()

def patched_get_chrome(self):
    # Get Chrome options correctly from the list
    options = self._options[0] if self._options else webdriver.ChromeOptions()
    # Set explicit ChromeDriver version
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

# Patch the _get_chrome method in dash.testing.browser.Browser.
dash.testing.browser.Browser._get_chrome = patched_get_chrome