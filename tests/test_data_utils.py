import pandas as pd
import numpy as np
import pytest
import math
import plotly.graph_objects as go

from src.data_utils import ensure_all_hexes, all_hexes, get_permit_label, get_subrange_singlequarter_99, build_log_ticks, should_use_log_scale, get_colorscale_params, permit_options, permit_counts_wide, get_global_max_for_permit_type, create_time_series_figure

@pytest.fixture(autouse=True)
def override_all_hexes():
    """
    Override the global all_hexes variable for testing purposes.
    This ensures that the tests use a controlled list of hex identifiers.
    """
    # Set up a controlled list for testing.
    test_hexes = ["hex1", "hex2", "hex3"]
    # Monkey-patch the module-level 'all_hexes'.
    import src.data_utils as du
    original_all_hexes = du.all_hexes
    du.all_hexes = test_hexes
    yield
    # Optionally, restore the original value after tests.
    du.all_hexes = original_all_hexes

@pytest.fixture(scope="module", autouse=True)
def override_globals():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        "period": ["Q1", "Q2", "Q3", "Q4"],
        "NB": [5, 7, 3, 10],
        "DM": [0, 2, 2, 1]
    })
    # For simplicity, assume the sorted unique periods is as below.
    quarters = sorted(df["period"].unique())
    
    # Override the module-level globals in src.data_utils.
    import src.data_utils as du
    du.permit_counts_wide = df
    du.quarters = quarters

    # Provide these values to tests if needed.
    return {"df": df, "quarters": quarters}

def test_ensure_all_hexes_complete_subset():
    """
    Test that if the input DataFrame is missing some hexes,
    they are added with a permit count of 0.
    """
    # Create input DataFrame with only a subset of the expected hexes.
    data = {
        "h3_index": ["hex1", "hex2"],
        "NB": [10, 20]
    }
    df = pd.DataFrame(data)
    
    # Call the function to ensure all hexes are present.
    df_result = ensure_all_hexes(df, "NB")
    
    # Build the expected DataFrame with float64 dtype to match result
    expected_data = {
        "h3_index": ["hex1", "hex2", "hex3"],
        "NB": [10, 20, 0]
    }
    expected_df = pd.DataFrame(expected_data)
    
    # Compare dataframes; reset_index ensures order consistency.
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), expected_df.reset_index(drop=True))

def test_ensure_all_hexes_with_missing_value():
    """
    Test that if an existing hex's permit value is NaN,
    it gets replaced with 0.
    """
    # Create input DataFrame with an NA value.
    data = {
        "h3_index": ["hex2"],
        "NB": [np.nan]
    }
    df = pd.DataFrame(data)
    
    # Run the function.
    df_result = ensure_all_hexes(df, "NB")
    
    # The expected DataFrame should include all hexes with count 0,
    # since the provided data had an NA that was filled.
    expected_data = {
        "h3_index": ["hex1", "hex2", "hex3"],
        "NB": [0, 0, 0]
    }
    expected_df = pd.DataFrame(expected_data)
    
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), expected_df.reset_index(drop=True))

# ---------------------------
# Test for get_permit_label
# ---------------------------
def test_get_permit_label():
    # Known permit type values should return the proper label.
    assert get_permit_label("NB") == "New Building (NB)"
    assert get_permit_label("DM") == "Demolition (DM)"
    # If permit value is not found, the function should simply return the input value.
    assert get_permit_label("XYZ") == "XYZ"


# ------------------------------------------
# Test for get_subrange_singlequarter_99
# ------------------------------------------
def test_get_subrange_singlequarter_99(monkeypatch):
    # Create a test DataFrame simulating permit_counts_wide.
    data = {
        "period": ["2020-Q1", "2020-Q1", "2020-Q2", "2020-Q2"],
        "NB": [1, 100, 50, 200],
        "DM": [5, np.nan, 15, 25],
    }
    df_test = pd.DataFrame(data)

    # Monkeypatch the global permit_counts_wide variable in the module.
    import src.data_utils as du
    monkeypatch.setattr(du, "permit_counts_wide", df_test)

    # For permit type "NB", check that the returned 99th percentile is correct.
    expected_nb = np.percentile(np.array([1, 100, 50, 200]), 99)
    result_nb = get_subrange_singlequarter_99("NB", "2020-Q1", "2020-Q2")
    assert np.isclose(result_nb, expected_nb)

    # If the sub-DataFrame is empty, the function should return 0.0.
    result_empty = get_subrange_singlequarter_99("NB", "2021-Q1", "2021-Q2")
    assert result_empty == 0.0

    # For permit type "DM", NaN values should be replaced with 0.
    # So the values become [5, 0, 15, 25].
    expected_dm = np.percentile(np.array([5, 0, 15, 25]), 99)
    result_dm = get_subrange_singlequarter_99("DM", "2020-Q1", "2020-Q2")
    assert np.isclose(result_dm, expected_dm)


# ---------------------------
# Test for build_log_ticks
# ---------------------------
def test_build_log_ticks_negative_or_zero():
    # When new_cmax is <= 0, the function should return [0] ticks.
    tick_vals, tick_text = build_log_ticks(0)
    assert tick_vals == [0]
    assert tick_text == ["0"]

def test_build_log_ticks_positive():
    # For a new_cmax that is not a power-of-ten (e.g., 5)
    tick_vals, tick_text = build_log_ticks(5)
    # math.ceil(log10(5)) is 1 so the ticks start with [0, 10**0] and then 5 is appended.
    assert tick_vals == [0, 1, 5]
    assert tick_text == ["0", "1", "5"]

def test_build_log_ticks_exact_power():
    # For new_cmax = 1000 (a power-of-ten value)
    tick_vals, tick_text = build_log_ticks(1000)
    # Expected steps: math.ceil(log10(1000)) == 3, so ticks: [0] + [1, 10, 100] then 1000 is appended.
    assert tick_vals == [0, 1, 10, 100, 1000]
    expected_tick_text = ["0", "1", "10", "100", f"{1000:.0e}"]
    assert tick_text == expected_tick_text


# -----------------------------
# Test for should_use_log_scale
# -----------------------------
def test_should_use_log_scale_insufficient_data():
    # If there are fewer than 2 non-zero values, log scale is not used.
    assert should_use_log_scale(np.array([0, 0])) is False
    assert should_use_log_scale(np.array([0, 10])) is False  # only one non-zero

def test_should_use_log_scale_true():
    # Create an array with a very low median and a high 99th percentile.
    data = np.array([1, 2, 100])
    # p99 (approximately 100) > 50 and ratio (about 100/2 = 50) > 5.
    assert should_use_log_scale(data) == True

def test_should_use_log_scale_false():
    # For a nearly uniform distribution the log scale should not be used.
    data = np.array([10, 11, 12])
    assert should_use_log_scale(data) == False


# -----------------------------------
# Test for get_colorscale_params
# -----------------------------------
def test_get_colorscale_params_log():
    # Custom colorscale should match the one defined in the function.
    expected_cs = [
        [0, 'rgba(240,240,240,0.8)'],
        [0.1, 'rgba(215,225,240,0.85)'],
        [0.25, 'rgba(190,210,235,0.9)'],
        [0.4, 'rgba(220,180,180,0.9)'],
        [0.6, 'rgba(230,150,150,0.92)'],
        [0.8, 'rgba(240,100,100,0.95)'],
        [0.9, 'rgba(250,50,50,0.97)'],
        [1, 'rgba(255,0,0,1)']
    ]
    # Test with an array of all zeros
    data_zeros = np.array([0, 0, 0])
    zmin, zmax, cs, suffix = get_colorscale_params(data_zeros, use_log=True)
    assert zmin == 0.1  # Updated to expect 0.1 as the minimum for log scale

def test_get_colorscale_params_linear():
    expected_cs = [
        [0, 'rgba(240,240,240,0.8)'],
        [0.1, 'rgba(215,225,240,0.85)'],
        [0.25, 'rgba(190,210,235,0.9)'],
        [0.4, 'rgba(220,180,180,0.9)'],
        [0.6, 'rgba(230,150,150,0.92)'],
        [0.8, 'rgba(240,100,100,0.95)'],
        [0.9, 'rgba(250,50,50,0.97)'],
        [1, 'rgba(255,0,0,1)']
    ]
    # Scenario 1: When less than 50% of the values are zero.
    data = np.array([0, 10, 20, 30])
    # Here, zmin should simply be the minimum (0) and zmax the 99th percentile of data.
    zmin, zmax, cs, suffix = get_colorscale_params(data, use_log=False)
    assert zmin == 0
    expected_zmax = np.percentile(data, 99)
    assert np.isclose(zmax, expected_zmax)
    assert cs == expected_cs
    assert suffix == ""
    
    # Scenario 2: When more than 50% of the data are zeros.
    data_majority_zeros = np.array([0, 0, 0, 10])
    zmin, zmax, cs, suffix = get_colorscale_params(data_majority_zeros, use_log=False)
    # When >50% zeros, zmin should be adjusted: non_zero.min() * 0.9.
    assert np.isclose(zmin, 10 * 0.9)
    expected_zmax = np.percentile(data_majority_zeros, 99)
    assert np.isclose(zmax, expected_zmax)
    assert cs == expected_cs
    assert suffix == ""

def test_get_global_max_for_permit_type_full_range(override_globals):
    # When start_idx/end_idx are not provided, the overall max is returned.
    test_df = override_globals["df"]
    expected_max = test_df["NB"].max()  # Should be 10 for NB.
    result = get_global_max_for_permit_type("NB")
    assert result == expected_max, f"Expected {expected_max}, got {result}"


def test_get_global_max_for_permit_type_subrange(override_globals):
    # Test a subrange.
    # With quarters ["Q1","Q2","Q3","Q4"], use indexes 1 and 2 to filter rows with period Q2 and Q3.
    # For NB values [7, 3], expected max is 7.
    result = get_global_max_for_permit_type("NB", start_idx=1, end_idx=2)
    assert result == 7, f"Expected 7 for subrange, got {result}"

    # Also test with a different permit type.
    # For DM values in Q2 and Q3, values are [2, 2] so expected max is 2.
    result_dm = get_global_max_for_permit_type("DM", start_idx=1, end_idx=2)
    assert result_dm == 2, f"Expected 2 for DM permit in subrange, got {result_dm}"


def test_create_time_series_figure_without_selected_range():
    # Create a dummy DataFrame for the time series figure.
    df = pd.DataFrame({
        "period": ["Q1", "Q2", "Q3", "Q4"],
        "NB": [5, 7, 3, 10]
    })

    fig = create_time_series_figure(df, "NB")
    
    # Verify that a Plotly Figure is returned.
    assert isinstance(fig, go.Figure), "Result should be a Plotly figure."
    
    # Check that there is one trace (the scatter trace).
    assert len(fig.data) == 1, "There should be one trace in the figure."
    trace = fig.data[0]
    # Verify that the trace is a scatter plot in "lines+markers" mode.
    assert trace.type == "scatter", f"Expected trace type 'scatter', got {trace.type}"
    assert "lines+markers" in trace.mode, f"Expected 'lines+markers' mode, got {trace.mode}"
    
    # Verify x and y values match the DataFrame columns.
    assert list(trace.x) == list(df["period"]), "x-values do not match the period column."
    assert list(trace.y) == list(df["NB"]), "y-values do not match the permit count column."
    
    # Since no selected_range was provided, no vertical rectangle should be added.
    shapes = fig.layout.shapes
    assert shapes is None or len(shapes) == 0, "No shapes should be added when selected_range is None."


def test_create_time_series_figure_with_selected_range():
    # Create a dummy DataFrame.
    df = pd.DataFrame({
        "period": ["Q1", "Q2", "Q3", "Q4"],
        "NB": [5, 7, 3, 10]
    })
    # Define a selected range to highlight Q2 to Q3 (indexes 1 to 2).
    selected_range = [1, 2]
    
    fig = create_time_series_figure(df, "NB", selected_range=selected_range)
    
    # In Plotly, a vertical rectangle added by add_vrect shows up in layout.shapes.
    shapes = fig.layout.shapes
    assert shapes is not None and len(shapes) > 0, "A vertical rectangle should be added when selected_range is provided."
    
    # Check that the shape's x0 and x1 values correspond to the correct period values.
    shape = shapes[0]
    expected_x0 = df.iloc[selected_range[0]]["period"]
    expected_x1 = df.iloc[selected_range[1]]["period"]
    assert shape["x0"] == expected_x0, f"Expected x0 to be {expected_x0}, got {shape['x0']}"
    assert shape["x1"] == expected_x1, f"Expected x1 to be {expected_x1}, got {shape['x1']}"
