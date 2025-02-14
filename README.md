# NYC Building Permit Dashboard

An interactive data science dashboard for visualizing and analyzing NYC building permit data. This project leverages modern web development and data science techniques to provide dynamic insights into permit issuance over time and across the city's geospatial landscape.

[![Python Version](https://img.shields.io/badge/python->=3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The **NYC Building Permit Dashboard** is built using [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/) to create a responsive, highly interactive web application. It processes geospatial data with [GeoPandas](https://geopandas.org/) and performs advanced data manipulation with [pandas](https://pandas.pydata.org/). By combining these powerful tools, the dashboard enables users to:

- Explore building permit trends over time via intuitive time-series visualizations.
- Interactively filter and analyze permits by type using real-time Dash callbacks.
- View spatial distributions of permits on NYC maps with aggregated and quarterly views.
- Leverage best engineering practices with a modular codebase and clean separation of concerns.

This application not only demonstrates proficiency in data science but also showcases robust web development skills.

---

## Features

- **Interactive Time-Series Analysis:** Visualize permit issuance trends with dynamic annotations for selected time ranges.
- **Choropleth Mapping:** Explore detailed and aggregated maps of NYC building permits using a hexagonal spatial grid.
- **Dynamic Filtering:** Easily switch between permit types and time periods with radio buttons, sliders, and dropdown menus.
- **Modular Design:** Clean separation between layout, callbacks, configuration, and data processing for maintainability and scalability.
- **Logging & Debugging:** Built-in logging and a hidden debug panel to facilitate troubleshooting during development.

---

## Installation

### Prerequisites

- Python >= 3.13
- [pip](https://pip.pypa.io/)
- (Optional) A virtual environment tool such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/).

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/nyc-building-permit-dashboard.git
   cd nyc-building-permit-dashboard
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   The dependencies are managed via a `pyproject.toml` file. Install them using [pip]:

   ```bash
   pip install -e .
   ```

   Alternatively, if you use a requirements file generated from your project configuration, run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Data Files:**

   Ensure the processed data is located in the `data/processed` directory. This data includes:
   - `nyc_hexes.geojson` – the geospatial grid for NYC
   - `permits_wide.csv` – the permit counts data

---

## Usage

To launch the dashboard locally, run:
```bash
python src/app.py
```

The app will start on [http://127.0.0.1:8050](http://127.0.0.1:8050). Open this link in your browser to interact with the dashboard.

---

## Project Structure

```
.
├── assets
│   ├── background.png         # Image assets
│   └── style.css              # Custom CSS styles
├── codebase.md                # Project tree view and documentation
├── data
│   └── processed              # Preprocessed geospatial and permit data
│       ├── nyc_hexes.geojson
│       └── permits_wide.csv
├── pyproject.toml             # Project configuration and dependency management
├── run_codeweaver.bat         # Batch script for documentation generation (Windows)
├── src
│   ├── app.py                 # Application entry point
│   ├── app_instance.py        # Dash app and server instance setup
│   ├── callbacks.py           # All interactive component callbacks
│   ├── config.py              # Configuration settings
│   ├── data_utils.py          # Data processing and visualization utility functions
│   ├── debug.py               # Debugging tools
│   └── layout.py              # Dashboard layout definitions
└── uv.lock                    # Dependency lock file
```

---

## Technology Stack

- **Python** – Core programming language.
- **Dash & Plotly** – For creating interactive web-based visualizations.
- **GeoPandas & Pandas** – For data manipulation and geospatial analysis.
- **Dash Bootstrap Components** – For responsive and attractive UI components.
- **Uvicorn & ASGI** – (Optional) For deploying the app in an asynchronous production setting.

---

## Data Science & Development Highlights

- **Advanced Data Manipulation:** Efficient aggregation and transformation of permit data, including time-series grouping and dynamic geospatial mapping.
- **Interactive Visualization:** Real-time updates across multiple charts and maps with advanced Dash callbacks.
- **Modular and Scalable Codebase:** Leveraging best practices in software engineering by separating concerns into distinct modules (data utilities, layout, callbacks, configuration).
- **Logging & Debugging:** Comprehensive logging throughout the data processing pipeline, facilitating easier identification and resolution of issues.
- **Deployment Ready:** Configured for local debugging with potential for asynchronous deployment using ASGI and Uvicorn.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/nyc-building-permit-dashboard/issues).

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

Created by [David Leather](https://daveleather.com). Explore more of my work and connect with me on [GitHub](https://github.com/dleather).

---

Happy coding!
