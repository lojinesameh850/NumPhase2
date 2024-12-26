import flet as ft
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import threading
import webview
import numpy as np
import queue

# Queue for communication between Flet and Dash
update_queue = queue.Queue()

# Global variable to manage WebView state
webview_window = None
webview_open = False

# Function to update the chart data dynamically
def generate_chart_data(custom_function):
    x = np.linspace(-10, 10, 100)
    y = custom_function(x)
    fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines", name="Custom Function")])
    fig.update_layout(title="Custom Function Plot")
    return fig

# Function to start the Dash app
def start_dash_app():
    app = dash.Dash(__name__)
    app.title = "Dynamic Function Plot"

    # Initial empty figure
    app.layout = html.Div(children=[
        html.H1("Dynamic Function Plot", style={"text-align": "center"}),
        dcc.Graph(id="graph", figure=go.Figure()),
        dcc.Interval(id="interval", interval=1000, n_intervals=0)  # Check for updates every second
    ])

    # Callback to update the chart when new data is available
    @app.callback(
        Output("graph", "figure"),
        [Input("interval", "n_intervals")]
    )
    def update_chart(n):
        try:
            # Check if a new function is in the queue
            custom_function = update_queue.get_nowait()
            return generate_chart_data(custom_function)
        except queue.Empty:
            # No update needed
            return dash.no_update

    app.run_server(debug=False, port=8050, use_reloader=False)

# Function to start or redisplay the WebView
def start_or_redisplay_webview():
    global webview_window, webview_open

    if not webview_open:
        webview_open = True  # Mark the window as open
        threading.current_thread().name = "MainThread"
        webview_window = webview.create_window("Plotly Chart", "http://localhost:8050", width=800, height=600)
        webview.start()
        webview_open = False  # Mark the window as closed after WebView exits

# Function to start the Flet app
def start_flet():
    def main(page: ft.Page):
        page.title = "Flet with Custom Plotly Chart"
        page.vertical_alignment = ft.MainAxisAlignment.CENTER

        # Function to create the graph based on user input
        def create_graph(e):
            func_str = function_input.value.strip()

            try:
                # Convert the string to a Python function
                custom_function = eval(f"lambda x: {func_str}")

                # Put the new function in the queue for Dash to process
                update_queue.put(custom_function)

                # Ensure the WebView window is displayed
                threading.Thread(target=start_or_redisplay_webview, daemon=True).start()

            except Exception as ex:
                result.value = f"Error in function: {str(ex)}"
            page.update()

        # Text field for function input
        function_input = ft.TextField(label="Enter a function (e.g., 'x**2', 'np.sin(x)')", expand=True)

        # Label to show errors or messages
        result = ft.Text()

        # Button to generate the graph
        generate_button = ft.ElevatedButton("Generate Graph", on_click=create_graph)

        # Add components to the Flet page
        page.add(
            function_input,
            generate_button,
            result
        )

    ft.app(target=main)

if __name__ == "__main__":
    # Start the Dash server in a separate thread
    threading.Thread(target=start_dash_app, daemon=True).start()

    start_flet()
