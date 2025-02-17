from src.callbacks import aggregator_callback, debug_map_selections, update_titles, reset_time_range, update_map_view, build_single_choropleth_map
import pytest
from dash import Dash
from dash.testing.composite import DashComposite
from unittest.mock import patch
import pandas as pd
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from src import callbacks
from dash import no_update
import math
import numpy as np

# A simple dummy callback context to simulate dash.callback_context.triggered
class DummyCallbackContext:
    def __init__(self, triggered):
        self.triggered = triggered

# Fixture for an initial global_filter state
@pytest.fixture
def initial_filter():
    return {
        "startQuarterIndex": 0,
        "endQuarterIndex": 2,
        "currentQuarterIndex": 1,
        "permitType": "NB",
        "play": False,
        "speed": 1000,
        "selectedHexes": []
    }

# Fixture for quarterly selection tests (for map-quarterly trigger)
@pytest.fixture
def quarterly_data(monkeypatch):
    # Define a simple set of quarters and a DataFrame for a single quarter ("Q1")
    quarters = ["Q1", "Q2", "Q3"]
    df = pd.DataFrame({
        "period": ["Q1", "Q1", "Q1"],
        "h3_index": ["a", "b", "c"],
        "NB": [10, 20, 30]
    })
    from src import data_utils
    monkeypatch.setattr(data_utils, "quarters", quarters)
    monkeypatch.setattr(data_utils, "permit_counts_wide", df)
    return quarters, df

# Fixture for aggregated map selection tests
@pytest.fixture
def aggregated_data(monkeypatch):
    # Define quarters and a DataFrame spanning multiple quarters.
    quarters = ["Q1", "Q2", "Q3"]
    df = pd.DataFrame({
        "period": ["Q1", "Q1", "Q2", "Q3", "Q3"],
        "h3_index": ["a", "b", "a", "b", "c"],
        "NB": [1, 2, 3, 4, 5]
    })
    from src import data_utils
    monkeypatch.setattr(data_utils, "quarters", quarters)
    monkeypatch.setattr(data_utils, "permit_counts_wide", df)
    return quarters, df

def test_aggregator_callback(monkeypatch):
    # Create a mock callback context
    class MockCallbackContext:
        def __init__(self):
            # Change trigger to slider
            self.triggered = [{"prop_id": "period-range-slider.value"}]

    # Patch the callback context
    monkeypatch.setattr('dash.callback_context', MockCallbackContext())

    # Set up sample inputs
    slider_value = [0, 5]
    permit_value = "NB"
    play_clicks = 1
    pause_clicks = 0
    speed_value = 1000
    n_intervals = 0
    qtr_sel = None
    agg_sel = None
    clear_hexes_clicks = None
    clear_time_clicks = None
    global_filter = {
        "startQuarterIndex": 0,
        "endQuarterIndex": 10,
        "currentQuarterIndex": 0,
        "permitType": "NB",
        "play": False,
        "speed": 1000,
        "selectedHexes": []
    }

    # Call the aggregator callback
    new_filter = aggregator_callback(
        slider_value,
        permit_value,
        play_clicks,
        pause_clicks,
        speed_value,
        n_intervals,
        qtr_sel,
        agg_sel,
        clear_hexes_clicks,
        clear_time_clicks,
        global_filter
    )

    # Perform assertions
    assert new_filter["startQuarterIndex"] == slider_value[0]
    assert new_filter["endQuarterIndex"] == slider_value[1]
    assert new_filter["permitType"] == permit_value

def test_debug_map_selections():
    """Test the debug_map_selections callback functionality"""
    
    # Test case 1: Basic selection data
    with patch('src.callbacks.logger') as mock_logger:
        quarterly_selection = {
            "points": [
                {"pointIndex": 0},
                {"pointIndex": 1}
            ]
        }
        aggregated_selection = {
            "points": [
                {"pointIndex": 2}
            ]
        }
        
        result = debug_map_selections(quarterly_selection, aggregated_selection)
        
        # Verify the return value
        assert result == ""
        
        # Verify logging calls
        mock_logger.info.assert_any_call("=== debug_map_selections fired ===")
        mock_logger.info.assert_any_call("Quarterly selection: %s", quarterly_selection)
        mock_logger.info.assert_any_call("Aggregated selection: %s", aggregated_selection)

    # Test case 2: Empty selections
    with patch('src.callbacks.logger') as mock_logger:
        result = debug_map_selections(None, None)
        
        assert result == ""
        mock_logger.info.assert_any_call("Quarterly selection: %s", None)
        mock_logger.info.assert_any_call("Aggregated selection: %s", None)

    # Test case 3: Mixed selections (one None, one with data)
    with patch('src.callbacks.logger') as mock_logger:
        quarterly_selection = {
            "points": [
                {"pointIndex": 0}
            ]
        }
        
        result = debug_map_selections(quarterly_selection, None)
        
        assert result == ""
        mock_logger.info.assert_any_call("Quarterly selection: %s", quarterly_selection)
        mock_logger.info.assert_any_call("Aggregated selection: %s", None)

    # Test case 4: Empty point lists
    with patch('src.callbacks.logger') as mock_logger:
        empty_selection = {"points": []}
        
        result = debug_map_selections(empty_selection, empty_selection)
        
        assert result == ""
        mock_logger.info.assert_any_call("Quarterly selection: %s", empty_selection)
        mock_logger.info.assert_any_call("Aggregated selection: %s", empty_selection)

@pytest.mark.integration
def test_debug_map_selections_prevent_initial_call(dash_duo):
    """Test that prevent_initial_call works as expected"""
    app = Dash(__name__)
    
    # Create a minimal layout that includes the callback inputs and outputs
    app.layout = html.Div([
        html.Div(id="dummy-output"),
        dcc.Graph(id="map-quarterly"),
        dcc.Graph(id="map-aggregated")
    ])
    
    # Register the callback
    app.callback(
        Output("dummy-output", "children"),
        [Input("map-quarterly", "selectedData"), 
         Input("map-aggregated", "selectedData")],
        prevent_initial_call=True
    )(debug_map_selections)
    
    # Start the test
    dash_duo.start_server(app)
    
    # Verify that the dummy-output is empty on initial load
    assert dash_duo.find_element("#dummy-output").text == ""

# 1) Test slider change: update time range from a slider input
def test_slider_change(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "period-range-slider.value"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)

    slider_value = [1, 2]  # new start and end indices
    result = aggregator_callback(
        slider_value,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["startQuarterIndex"] == 1
    assert result["endQuarterIndex"] == 2
    # currentQuarterIndex is already 1 and within [1,2]
    assert result["currentQuarterIndex"] == 1

# 1a) Test slider change when the currentQuarterIndex falls outside the new slider range
def test_slider_change_current_clamp(monkeypatch, initial_filter, aggregated_data):
    initial_filter["currentQuarterIndex"] = 0
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "period-range-slider.value"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    slider_value = [1, 2]
    result = aggregator_callback(
        slider_value,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    # currentQuarterIndex should now be clamped to the start value, 1
    assert result["startQuarterIndex"] == 1
    assert result["endQuarterIndex"] == 2
    assert result["currentQuarterIndex"] == 1

# 2) Test permit type change
def test_permit_type_change(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "permit-type.value"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    permit_value = "DM"  # Change permit type from "NB" to "DM"
    result = aggregator_callback(
        slider_value=None,
        permit_value=permit_value,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["permitType"] == "DM"

# 3) Test play-button trigger: sets "play" to True
def test_play_button(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "play-button.n_clicks"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=1,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["play"] is True

# 4) Test pause-button trigger: sets "play" to False
def test_pause_button(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "pause-button.n_clicks"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=1,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["play"] is False

# 5) Test speed change trigger: updates global_filter["speed"]
def test_speed_change(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "speed-radio.value"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    speed_value = 2000
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=speed_value,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["speed"] == speed_value

# 6) Test animation interval tick: advancing the current quarter with wrapping
def test_animation_interval(monkeypatch, initial_filter, aggregated_data):
    quarters, _ = aggregated_data
    # Set currentQuarterIndex at the end so that the tick wraps around
    initial_filter["currentQuarterIndex"] = 2
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "animation-interval.n_intervals"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=1,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    # Since currentQuarterIndex (2) plus one exceeds endQuarterIndex (2),
    # it should wrap to startQuarterIndex which is 0.
    assert result["currentQuarterIndex"] == 0

# 6a) Test a regular animation tick increment (without wrapping)
def test_animation_interval_regular(monkeypatch, initial_filter, aggregated_data):
    initial_filter["currentQuarterIndex"] = 1
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "animation-interval.n_intervals"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=1,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    # currentQuarterIndex should increment by 1, from 1 to 2
    assert result["currentQuarterIndex"] == 2

# 7) Test map-quarterly selection: update "selectedHexes" based on quarterly map points
def test_map_quarterly_selection(monkeypatch, initial_filter, quarterly_data):
    quarters, df = quarterly_data
    # Ensure currentQuarterIndex points to "Q1"
    initial_filter["currentQuarterIndex"] = 0
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "map-quarterly.selectedData"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    # Simulate selection with one point - pointIndex 1 in the DataFrame sorted by h3_index ["a", "b", "c"]
    qtr_sel = {"points": [{"pointIndex": 1}]}
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=qtr_sel,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    # Expect "b" based on the ordering in the quarterly DataFrame.
    assert result["selectedHexes"] == ["b"]

# 8) Test map-aggregated selection: update "selectedHexes" based on aggregated map points
def test_map_aggregated_selection(monkeypatch, initial_filter, aggregated_data):
    quarters, df = aggregated_data
    # Set the time range to cover all quarters and permitType to "NB"
    initial_filter["startQuarterIndex"] = 0
    initial_filter["endQuarterIndex"] = 2
    initial_filter["permitType"] = "NB"
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "map-aggregated.selectedData"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    # Simulate selection: choose the third point (pointIndex 2)
    agg_sel = {"points": [{"pointIndex": 2}]}
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=agg_sel,
        clear_hexes_clicks=None,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    # Grouping produces a sorted order of h3_index: ["a", "b", "c"], so selection returns "c".
    assert result["selectedHexes"] == ["c"]

# 9) Test clear-hexes trigger: resets "selectedHexes" to an empty list
def test_clear_hexes(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "clear-hexes.n_clicks"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    # Assume some hexes are already selected
    initial_filter["selectedHexes"] = ["a", "b"]
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=1,
        clear_time_clicks=None,
        global_filter=initial_filter
    )
    assert result["selectedHexes"] == []

# 10) Test clear-time-range trigger: resets time range and current quarter
def test_clear_time_range(monkeypatch, initial_filter, aggregated_data):
    dummy_ctx = DummyCallbackContext(triggered=[{"prop_id": "clear-time-range.n_clicks"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)
    # Set non-default time range and current index
    initial_filter["startQuarterIndex"] = 1
    initial_filter["endQuarterIndex"] = 2
    initial_filter["currentQuarterIndex"] = 2
    quarters, _ = aggregated_data
    result = aggregator_callback(
        slider_value=None,
        permit_value=None,
        play_clicks=None,
        pause_clicks=None,
        speed_value=None,
        n_intervals=None,
        qtr_sel=None,
        agg_sel=None,
        clear_hexes_clicks=None,
        clear_time_clicks=1,
        global_filter=initial_filter
    )
    assert result["startQuarterIndex"] == 0
    assert result["endQuarterIndex"] == len(quarters) - 1
    assert result["currentQuarterIndex"] == 0

# ---------------------------
# Fixtures for dummy data
# ---------------------------
@pytest.fixture
def dummy_data_quarterly(monkeypatch):
    # Define a known set of quarters and a small dataframe for the quarterly map
    quarters = ["Q1", "Q2", "Q3"]
    df = pd.DataFrame({
        "period": ["Q1", "Q1", "Q2"],
        "h3_index": ["a", "b", "a"],
        "NB": [10, 20, 30]
    })
    # Override globals in the callbacks module
    monkeypatch.setattr(callbacks, "quarters", quarters)
    monkeypatch.setattr(callbacks, "permit_counts_wide", df)
    # Patch get_global_max_for_permit_type to return a fixed value (e.g., 20)
    monkeypatch.setattr(callbacks, "get_global_max_for_permit_type", lambda permit, s, e: 20)
    return quarters, df

@pytest.fixture
def dummy_data_aggregated(monkeypatch):
    # Define a known set of quarters and a small dataframe spanning multiple quarters
    quarters = ["Q1", "Q2", "Q3"]
    df = pd.DataFrame({
        "period": ["Q1", "Q1", "Q2", "Q3", "Q3"],
        "h3_index": ["a", "b", "a", "b", "c"],
        "NB": [1, 2, 3, 4, 5]
    })
    monkeypatch.setattr(callbacks, "quarters", quarters)
    monkeypatch.setattr(callbacks, "permit_counts_wide", df)
    return quarters, df

@pytest.fixture
def dummy_data_time_series(monkeypatch):
    # Dummy data for time-series
    quarters = ["Q1", "Q2", "Q3"]
    df = pd.DataFrame({
        "period": ["Q1", "Q1", "Q2", "Q3"],
        "h3_index": ["a", "b", "a", "c"],
        "NB": [5, 10, 20, 15]
    })
    monkeypatch.setattr(callbacks, "quarters", quarters)
    monkeypatch.setattr(callbacks, "permit_counts_wide", df)
    # For grouping by quarter, we need quarter_to_index.
    monkeypatch.setattr(callbacks, "quarter_to_index", lambda x: quarters.index(x) if x in quarters else None)
    return quarters, df

# -------------------------------------
# Test for dcc.Interval control callback
# -------------------------------------
def test_control_animation_interval_play():
    # When play is True the interval should be enabled (i.e. disabled==False)
    global_filter = {"play": True, "speed": 500}
    result = callbacks.control_animation_interval(global_filter)
    # Disabled should be the opposite of play, and speed passed unchanged
    assert result == (False, 500)

def test_control_animation_interval_pause():
    # When play is False the interval is disabled (i.e. disabled==True)
    global_filter = {"play": False, "speed": 1500}
    result = callbacks.control_animation_interval(global_filter)
    assert result == (True, 1500)

# -------------------------------------
# Tests for the Quarterly Map callback
# -------------------------------------
def test_update_quarterly_map_no_mapview(dummy_data_quarterly):
    quarters, _ = dummy_data_quarterly
    global_filter = {
        "permitType": "NB",
        "currentQuarterIndex": 0,  # Maps to quarter "Q1"
        "selectedHexes": ["a"],
        "startQuarterIndex": 0,
        "endQuarterIndex": len(quarters) - 1
    }
    map_view = None  # No map view override
    fig = callbacks.update_quarterly_map(global_filter, map_view)
    
    # Verify that a Plotly Figure is returned
    assert isinstance(fig, go.Figure)
    
    # Instead of comparing directly (which returns a tuple), convert to list.
    selected_points = None
    for trace in fig.data:
        if hasattr(trace, "selectedpoints") and trace.selectedpoints is not None:
            selected_points = trace.selectedpoints
    assert list(selected_points) == [0]

def test_update_quarterly_map_with_mapview(dummy_data_quarterly):
    quarters, _ = dummy_data_quarterly
    global_filter = {
        "permitType": "NB",
        "currentQuarterIndex": 0,  # "Q1"
        "selectedHexes": [],
        "startQuarterIndex": 0,
        "endQuarterIndex": len(quarters) - 1
    }
    map_view = {
        "center": {"lat": 40.0, "lon": -74.0},
        "zoom": 11,
        "bearing": 10,
        "pitch": 0
    }
    fig = callbacks.update_quarterly_map(global_filter, map_view)
    layout = fig.layout
    center = layout.mapbox.center
    assert center.lat == map_view["center"]["lat"]
    assert center.lon == map_view["center"]["lon"]

# -------------------------------------
# Tests for the Aggregated Map callback
# -------------------------------------
def test_update_aggregated_map_no_mapview(dummy_data_aggregated):
    quarters, _ = dummy_data_aggregated
    global_filter = {
       "permitType": "NB",
       "startQuarterIndex": 0,
       "endQuarterIndex": len(quarters) - 1,
       "selectedHexes": ["b"]
    }
    map_view = None
    fig = callbacks.update_aggregated_map(global_filter, map_view)
    assert isinstance(fig, go.Figure)
    
    selected_points = None
    for trace in fig.data:
        if hasattr(trace, "selectedpoints") and trace.selectedpoints is not None:
            selected_points = trace.selectedpoints
    assert list(selected_points) == [1]

def test_update_aggregated_map_with_mapview(dummy_data_aggregated):
    quarters, _ = dummy_data_aggregated
    global_filter = {
       "permitType": "NB",
       "startQuarterIndex": 1,  # From "Q2"
       "endQuarterIndex": 2,      # to "Q3"
       "selectedHexes": []
    }
    map_view = {
        "center": {"lat": 41.0, "lon": -73.0},
        "zoom": 10,
        "bearing": 0,
        "pitch": 5
    }
    fig = callbacks.update_aggregated_map(global_filter, map_view)
    layout = fig.layout
    center = layout.mapbox.center
    assert center.lat == map_view["center"]["lat"]
    assert center.lon == map_view["center"]["lon"]

# -------------------------------------
# Tests for Time Series callback
# -------------------------------------
def test_update_time_series_with_current_line(dummy_data_time_series):
    quarters, _ = dummy_data_time_series
    global_filter = {
        "permitType": "NB",
        "startQuarterIndex": 0,
        "endQuarterIndex": len(quarters) - 1,
        "currentQuarterIndex": 1,  # "Q2"
        "selectedHexes": []
    }
    fig = callbacks.update_time_series(global_filter)
    assert isinstance(fig, go.Figure)
    
    # The callback is set to add a vertical dashed line for the current quarter
    shapes = fig.layout.shapes
    assert shapes is not None and len(shapes) > 0, "Expected a vertical line shape for the current quarter."
    
    # Check that the first shape is the expected vertical line at "Q2"
    line_shape = shapes[0]
    assert line_shape["type"] == "line"
    assert line_shape["x0"] == "Q2"
    assert line_shape["x1"] == "Q2"
    assert line_shape["line"]["color"] == "blue"
    assert line_shape["line"]["dash"] == "dash"

# -----------------------------------------------------------------------------
# Setup globals for testing
# -----------------------------------------------------------------------------
# Define a dummy list of quarters for testing.
test_quarters = ["Q1", "Q2", "Q3"]

# Patch the global variable "quarters" inside the callbacks module.
update_titles.__globals__["quarters"] = test_quarters
reset_time_range.__globals__["quarters"] = test_quarters

# Patch the get_permit_label function so that we control its output.
def fake_get_permit_label(permit_type):
    return f"Label for {permit_type}"

update_titles.__globals__["get_permit_label"] = fake_get_permit_label

# -----------------------------------------------------------------------------
# Tests for update_titles
# -----------------------------------------------------------------------------
def test_update_titles_default():
    """
    With no values in the global_filter, default values should be used:
      - permitType defaults to "NB"
      - startQuarterIndex defaults to 0 and endQuarterIndex defaults to len(quarters)-1.
    """
    global_filter = {}
    # Expected title breakdown:
    # permit_label = fake_get_permit_label("NB") -> "Label for NB"
    # start_label = test_quarters[0] -> "Q1"
    # end_label   = test_quarters[2] -> "Q3"
    expected_quarterly_title = "Label for NB Permits Issued Across Space and Time"
    expected_aggregated_title = "Label for NB Permits Issued from Q1 - Q3"
    expected_time_series_title  = "Time-Series of Label for NB"

    result = update_titles(global_filter)
    assert result == (expected_quarterly_title, expected_aggregated_title, expected_time_series_title)


def test_update_titles_custom():
    """
    When explicit values are provided in the global_filter they should be used.
    """
    global_filter = {
        "permitType": "ABC",
        "startQuarterIndex": 1,
        "endQuarterIndex": 1
    }
    # permit_label = fake_get_permit_label("ABC") -> "Label for ABC"
    # start_label and end_label will be test_quarters[1] -> "Q2"
    expected_quarterly_title = "Label for ABC Permits Issued Across Space and Time"
    expected_aggregated_title = "Label for ABC Permits Issued from Q2 - Q2"
    expected_time_series_title  = "Time-Series of Label for ABC"

    result = update_titles(global_filter)
    assert result == (expected_quarterly_title, expected_aggregated_title, expected_time_series_title)

# -----------------------------------------------------------------------------
# Tests for reset_time_range
# -----------------------------------------------------------------------------
def test_reset_time_range_clicked():
    """
    When the clear time range button is clicked (n_clicks truthy), we expect
    a reset slider range: [0, len(quarters) - 1]
    """
    result = reset_time_range(1)
    assert result == [0, len(test_quarters) - 1]


def test_reset_time_range_not_clicked():
    """
    For inputs where n_clicks is falsey (e.g., 0), the function should return no_update.
    """
    result = reset_time_range(0)
    assert result == dash.no_update


def test_reset_time_range_none():
    """
    If n_clicks is None (i.e. not clicked), the function should also return no_update.
    """
    result = reset_time_range(None)
    assert result == dash.no_update


# -----------------------------------------------------------------------------
# Tests for update_map_view
# -----------------------------------------------------------------------------
# We need to simulate dash.callback_context for these tests.
class DummyCtx:
    def __init__(self, triggered):
        self.triggered = triggered


def test_update_map_view_aggregated(monkeypatch):
    """
    If the callback is triggered by the aggregated map's relayoutData, then the new view
    is extracted from agg_relayout and merged into the current_view.
    """
    # Simulate that map-aggregated triggered the callback.
    dummy_ctx = DummyCtx(triggered=[{"prop_id": "map-aggregated.relayoutData"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)

    agg_relayout = {
        'map.center': {'lon': -73.0, 'lat': 40.0},
        'map.zoom': 10,
        'map.bearing': 15,
        'map.pitch': 5
    }
    qtr_relayout = None
    current_view = {"center": {'lon': 0, 'lat': 0}, "zoom": 5}

    expected_new_view = {
        "center": {'lon': -73.0, 'lat': 40.0},
        "zoom": 10,
        "bearing": 15,
        "pitch": 5
    }
    expected_result = current_view.copy()
    expected_result.update(expected_new_view)

    result = update_map_view(agg_relayout, qtr_relayout, current_view)
    assert result == expected_result


def test_update_map_view_quarterly(monkeypatch):
    """
    If the callback is triggered by the quarterly map's relayoutData, then the new view
    is extracted from qtr_relayout and merged into the current_view.
    """
    dummy_ctx = DummyCtx(triggered=[{"prop_id": "map-quarterly.relayoutData"}])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)

    agg_relayout = None
    qtr_relayout = {
        'map.center': {'lon': -74.0, 'lat': 41.0},
        'map.zoom': 9,
        'map.pitch': 0,
        'map.bearing': 0
    }
    current_view = None

    expected_new_view = {
        "center": {'lon': -74.0, 'lat': 41.0},
        "zoom": 9,
        "pitch": 0,
        "bearing": 0
    }

    result = update_map_view(agg_relayout, qtr_relayout, current_view)
    assert result == expected_new_view


def test_update_map_view_no_trigger(monkeypatch):
    """
    If no callback trigger is detected (i.e. an empty triggered list), then
    the function should simply return the current_view unchanged.
    """
    dummy_ctx = DummyCtx(triggered=[])
    monkeypatch.setattr(dash, "callback_context", dummy_ctx)

    agg_relayout = {
        'map.center': {'lon': -73.0, 'lat': 40.0},
        'map.zoom': 10,
    }
    qtr_relayout = {
        'map.center': {'lon': -74.0, 'lat': 41.0},
        'map.zoom': 9,
    }
    current_view = {"existing": "value"}

    result = update_map_view(agg_relayout, qtr_relayout, current_view)
    assert result == current_view

def test_build_single_choropleth_map_linear(monkeypatch):
    """
    Test the choropleth map construction using the linear scale branch.
    We supply a fixed zmax_override so that the computed zmax is predictable.
    """
    # Create a simple DataFrame
    df = pd.DataFrame({
        "h3_index": ["a", "b", "c"],
        "NB": [30, 40, 50]
    })
    permit_type = "NB"
    map_title = "Test Map Linear"
    zmax_override = 60  # force a known maximum for the linear branch

    # Force should_use_log_scale to always return False
    monkeypatch.setattr("src.callbacks.should_use_log_scale", lambda values: False)

    # Return a valid named colorscale (e.g. "blues") instead of the string "linear_colorscale"
    def fake_get_colorscale_params(values, use_log):
        return (None, None, "blues", "")
    monkeypatch.setattr("src.callbacks.get_colorscale_params", fake_get_colorscale_params)

    # Set the required global variables: hex_geojson and MAPBOX_STYLE.
    monkeypatch.setitem(build_single_choropleth_map.__globals__, "hex_geojson", {})
    monkeypatch.setitem(build_single_choropleth_map.__globals__, "MAPBOX_STYLE", "dummy_mapbox_style")

    # Call the function under test.
    fig = build_single_choropleth_map(df, permit_type, map_title, zmax_override=zmax_override)

    # Assert that a Figure is returned.
    assert isinstance(fig, go.Figure)

    # Check that the layout title and mapbox settings are set as expected.
    assert fig.layout.title.text == map_title
    assert fig.layout.mapbox.style == "dummy_mapbox_style"
    center = fig.layout.mapbox.center
    assert center.lat == 40.7
    assert center.lon == -73.9

    # There should be a single choroplethmapbox trace.
    assert len(fig.data) == 1
    trace = fig.data[0]
    # type check: the trace type (or its class) should be a Choroplethmapbox trace.
    # We check the locations and z data.
    assert list(trace.locations) == list(df["h3_index"])
    np.testing.assert_array_equal(trace.z, df["NB"])

    # Verify that the marker properties are correct.
    assert trace.marker.line.width == 1
    assert trace.marker.line.color == "rgba(255, 255, 255, 0.5)"

    # In the linear branch the tickvals and ticktext are set to None.
    cb = trace.colorbar
    assert cb.tickvals is None
    assert cb.ticktext is None
    # The colorbar title is built as "Count" + suffix; here suffix is an empty string.
    assert cb.title.text == "Count"


def test_build_single_choropleth_map_log(monkeypatch):
    """
    Test the choropleth map construction using the log scale branch.
    We simulate a dataset that should trigger log scaling and force the helper
    functions to return fixed dummy values.
    """
    # Create a DataFrame that should trigger log scaling.
    df = pd.DataFrame({
        "h3_index": ["a", "b", "c"],
        "NB": [1, 1, 100]
    })
    permit_type = "NB"
    map_title = "Test Map Log"

    # Force should_use_log_scale to always return True (log branch).
    monkeypatch.setattr("src.callbacks.should_use_log_scale", lambda values: True)

    # Return dummy values with a valid named colorscale "viridis"
    def fake_get_colorscale_params(values, use_log):
        return (1, 100, "viridis", " (log scale)")
    monkeypatch.setattr("src.callbacks.get_colorscale_params", fake_get_colorscale_params)

    # Monkey-patch build_log_ticks to return preset tick values and text.
    monkeypatch.setattr("src.callbacks.build_log_ticks", lambda log_val: ([1, 10], ["1", "10"]))

    monkeypatch.setitem(build_single_choropleth_map.__globals__, "hex_geojson", {})
    monkeypatch.setitem(build_single_choropleth_map.__globals__, "MAPBOX_STYLE", "dummy_mapbox_style")

    # Call the function under test.
    fig = build_single_choropleth_map(df, permit_type, map_title, zmax_override=None)

    # Continue with your assertions...
    assert isinstance(fig, go.Figure)
    # ... (other assertions)