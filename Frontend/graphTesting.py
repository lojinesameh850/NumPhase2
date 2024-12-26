import flet as ft
from sympy import symbols, sympify, lambdify
import numpy as np
import plotly.graph_objects as go
import webbrowser

def create_plot(func_str):
    try:
        x = symbols('x')
        
        func = sympify(func_str)
        
        func_num = lambdify(x, func, modules=["numpy"])
        
        x_vals = np.linspace(-200, 10, 500)
        y_vals = func_num(x_vals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'f(x) = {func_str}'))
        
        fig.update_layout(
            title="Graph of the Function",
            xaxis_title="x",
            yaxis_title="f(x)",
            template="plotly_white",
        )
        
        plot_file = "plot.html"
        fig.write_html(plot_file)
        
        return plot_file, None
    except Exception as e:
        return None, str(e)

def main(page: ft.Page):
    input_field = ft.TextField(label="Enter a function (e.g., sin(x) + x**2)", expand=True)
    
    def plot_function(event):
        plot_file, error = create_plot(input_field.value)
        if plot_file:
            # Open the saved graph in the default web browser
            webbrowser.open(plot_file)
        else:
            # Show error message
            page.snack_bar = ft.SnackBar(ft.Text(f"Error: {error}"))
            page.snack_bar.open = True
            page.update()
    
    plot_button = ft.ElevatedButton("Plot", on_click=plot_function)
    
    # Layout
    page.add(
        ft.Column(
            controls=[
                input_field,
                plot_button,
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=20,
        )
    )

ft.app(target=main)
