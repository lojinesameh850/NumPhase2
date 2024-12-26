import flet as ft
from functions import alphaBackend, send_to_backend, get_child, send_to_backend_root, handleSingleStepButtonClick, plot_function, send_to_backend_root_single_step

def addCells(size, matrix):
    for i in range(size):
        row = ft.Row()
        for j in range(size+1):
            cell = ft.TextField(width=50 , height=50)
            row.controls.append(cell)
            if(j<size-1):
                text = ft.Text(value=f"x{j+1}+")
                row.controls.append(text)
            elif(j==size-1):
                text1 = ft.Text(value=f"x{j+1} = ")
                row.controls.append(text1)
        matrix.controls.append(row)
def resize(event , page : ft.Column):
    try:
        update_suboptions(None , page)
    except:
        pass
    size_dropdown : ft.Dropdown = get_child(page , "size_dropdown")
    size = int(size_dropdown.value)
    matrix_container : ft.Column = get_child(page , "matrix_container")
    matrix_container.controls.clear()

    addCells(size , matrix_container)

    matrix_container.update()
def initialize_matrix(page : ft.Column):
    try:
        update_suboptions(None , page)
    except:
        pass
    matrix = get_child(page , "matrix_container")
    size = 2

    addCells(size , matrix)

    matrix.update()
def update_suboptions(event ,page :ft.Page ):
    operation = get_child(page , "operation_dropdown")
    suboptions = get_child(page , "suboptions")
    oper_type = int(operation.value)
    suboptions.controls.clear()
    significant_label = ft.Text(value="Significant Digits")
    significant_digits = ft.TextField(hint_text="Enter number of significant digits" , width=500 , key="significant_digits", value=4 , on_change= lambda e : e.control.focus() )
    if oper_type in {1,2}:
        suboptions.controls.append(ft.Column(controls=[significant_label , significant_digits] , key="significant_row"))
    elif oper_type in {3 , 4}:
        
        criteria1 = ft.TextField(hint_text="Enter max iterations" , key="its_text_field" , width=500, on_change= lambda e : e.control.focus())
        criteria2 = ft.TextField(hint_text="Enter relative error" , key="eps_text_field" , width=500, on_change= lambda e : e.control.focus())
        
        initial_guess_row = ft.Row(key="initial_guess_row")
        initial_guess_label = ft.Text(value="Initial Guess" , key="initial_guess_label")
        initial_guess_row.controls.append(initial_guess_label)
        size_dropdown : ft.Dropdown = get_child(page , "size_dropdown")
        size = int(size_dropdown.value)
        for i in range(size):
            field = ft.TextField(width=50, height=50 , on_change= lambda e : e.control.focus())
            initial_guess_row.controls.append(field)
        suboptions.controls.append(initial_guess_row)
        suboptions.controls.append(criteria1)
        suboptions.controls.append(criteria2)
        suboptions.controls.append(ft.Column(controls=[significant_label , significant_digits] , key="significant_row"))
    elif oper_type == 5:
        LU_sub_operations = ft.Dropdown(
            options=[
                ft.dropdown.Option("1" , "Doolittle Decomposition"),
                ft.dropdown.Option("2" , "Crout Decomposition"),
                ft.dropdown.Option("3" , "Cholesky Decomposition"),
            ],
            label= "Choose LU Sub Operation",
            width= 500,value="1",key="LU_sub"
        )
        suboptions.controls.append(LU_sub_operations)
        suboptions.controls.append(ft.Column(controls=[significant_label , significant_digits] , key="significant_row"))
    suboptions.update()

def update_suboptions_root(event, page : ft.Page ):
    # operation choosen decides what options are shown
    operation = get_child(page , "operation_dropdown_root")
    # the coloumn where we'll add the options
    suboptions_root = get_child(page , "suboptions_root")
    oper_type = int(operation.value)

    suboptions_root.controls.clear()

    significant_label = ft.Text(value="Significant Digits")
    significant_digits = ft.TextField(hint_text="Enter number of significant digits" , width=500 , key="significant_digits_root", value=15 , on_change= lambda e : e.control.focus() )
    suboptions_root.controls.append(ft.Column(controls=[significant_label , significant_digits] , key="significant_row_root"))

    epsilon_label = ft.Text(value="Tolerance (Epsilon)")
    epsilon = ft.TextField(hint_text="Enter the epsilon" , width=300 , key="epsilon_root", value=0.00001 , on_change= lambda e : e.control.focus() )
    suboptions_root.controls.append(ft.Column(controls=[epsilon_label , epsilon] , key="epsilon_row_root"))
    
    iterations_label = ft.Text(value="Max Iterations")
    iterations = ft.TextField(hint_text="Enter number of significant digits" , width=300 , key="iterations_root", value=50 , on_change= lambda e : e.control.focus() )
    suboptions_root.controls.append(ft.Column(controls=[iterations_label , iterations] , key="iterations_row_root"))
    
    # Bisection, false position has xl and xu
    if oper_type in {1, 2}:
        initial_interval_row = ft.Row(key="initial_interval_row")
        initial_interval_label = ft.Text(value="Initial Interval" , key="initial_interval_label")
        xl_field = ft.TextField(hint_text="xl", key="xl", width=80, height=50 , on_change= lambda e : e.control.focus())
        xu_field = ft.TextField(hint_text="xu", key="xu", width=80, height=50 , on_change= lambda e : e.control.focus())

        initial_interval_row.controls.append(initial_interval_label)
        initial_interval_row.controls.append(xl_field)
        initial_interval_row.controls.append(xu_field)

        suboptions_root.controls.append(initial_interval_row)

    # Fixed point, Original/Modified Newton-Raphson, secant x0 and possibly (Modified Newton only) m   
    elif oper_type in {3, 4, 5, 6}:

        # Modified Newton-Raphson x0, 1) Sol 1 : m, 2) Sol 2 : there's multiplicity but don't know m explicitly
        
        initial_x_row = ft.Row(key="initial_x_row")
        initial_x_label = ft.Text(value="Initial Guess" , key="initial_x_label")
        x0_field = ft.TextField(hint_text="x0", key="x0", width=100, height=50 , on_change= lambda e : e.control.focus())

        initial_x_row.controls.append(initial_x_label)
        initial_x_row.controls.append(x0_field)

        if oper_type == 6:
            x1_field = ft.TextField(hint_text="x-1", key="x-1", width=100, height=50 , on_change= lambda e : e.control.focus())
            initial_x_row.controls.append(x1_field)

        suboptions_root.controls.append(initial_x_row)

        


    suboptions_root.update()


def tab2():
    tab2 = ft.Column(key="Tab2")
    matrix_panel = ft.Column(key="matrix_container")
    size_dropdown = ft.Dropdown(
        options=
        [
            ft.dropdown.Option("2",2),
            ft.dropdown.Option("3",3),
        ],
        key="size_dropdown",
        value="2",
        width=500,
        on_change= lambda e : resize(e , tab2)
    )
    submit_button = ft.TextButton(
    text="Answer",
    key="submit_button",
    style=ft.ButtonStyle(
        shape=ft.RoundedRectangleBorder(radius=4),
        side=ft.BorderSide(color="blue", width=2),
        ),
        on_click = lambda e : alphaBackend(e , tab2)
    )
    tab2.controls.append(size_dropdown)
    tab2.controls.append(matrix_panel)
    tab2.controls.append(submit_button)
    return tab2
def tab1():
    tab1 = ft.Column(key="Tab1")
    matrix_panel = ft.Column(key="matrix_container")
    size_dropdown = ft.Dropdown(
        options=
        [
            ft.dropdown.Option("2",2),
            ft.dropdown.Option("3",3),
            ft.dropdown.Option("4",4),
            ft.dropdown.Option("5",5),
            ft.dropdown.Option("6",6),
            ft.dropdown.Option("7",7),
        ],
        label="Choose Size",
        key="size_dropdown",
        value="2",
        width=500,
        on_change= lambda e : resize(e , tab1)
    )
    tab1.controls.append(size_dropdown)
    tab1.controls.append(matrix_panel)
    
    operation_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("1" , "Gauss Elimination"),
            ft.dropdown.Option("2" , "Gauss-Jordan Elimination"),
            ft.dropdown.Option("3" , "Jacobi"),
            ft.dropdown.Option("4" , "Gauss-Seidel"),
            ft.dropdown.Option("5" , "LU Decomposition"),
        ],
        label="Choose Operation",
        value="1",
        width=500,
        key="operation_dropdown",
        on_change= lambda e : update_suboptions(e , tab1)
    )
    tab1.controls.append(operation_dropdown)
    suboptions = ft.Column(key="suboptions")
    tab1.controls.append(suboptions)
    submit_button = ft.TextButton(
    text="Answer",
    key="submit_button",
    style=ft.ButtonStyle(
        shape=ft.RoundedRectangleBorder(radius=4),
        side=ft.BorderSide(color="blue", width=2),
        ),
        on_click = lambda e : send_to_backend(e , tab1)
    )
    tab1.controls.append(submit_button)
    return tab1

def tab3():
    tab3 = ft.Column(key="Tab3")
    function_input_label = ft.Text(value="Enter the function", key="function_input_label")
    function_input = ft.TextField(
        hint_text = "eg: sin(2*x) - x**3",
        key = "funtion_input_string",
        on_change= lambda e : e.control.focus(), 
    )

    tab3.controls.append(ft.Column(controls=[function_input_label , function_input] , key="function_input_col"))    
    
    operation_dropdown = ft.Dropdown(
        options=[
            ft.dropdown.Option("1" , "Bisection"),
            ft.dropdown.Option("2" , "False-Position"),
            ft.dropdown.Option("3" , "Fixed Point"),
            ft.dropdown.Option("4" , "Original Newton-Raphson"),
            ft.dropdown.Option("5" , "Modified Newton-Raphson (Unknown Multiplicity)"),
            ft.dropdown.Option("6" , "Secant Method"),
        ],
        label="Choose Operation",
        value="1",
        width=500,
        key="operation_dropdown_root",
        on_change = lambda e : update_suboptions_root(e , tab3)
    )
    
    tab3.controls.append(operation_dropdown)

    suboptions = ft.Column(key="suboptions_root")

    tab3.controls.append(suboptions)


    graph_label = ft.Text(value="Enter the graph range", key="graph_label")
    graph_input_1 = ft.TextField(
        value = -50,
        key = "graph_input_1",
        width = 70,
        on_change= lambda e : e.control.focus(),
    )

    graph_input_2 = ft.TextField(
        value = 50,
        key = "graph_input_2",
        width = 70,
        on_change= lambda e : e.control.focus(),
    )

    graph_button = ft.TextButton(
        text="Show Graph",
        key="graph_button_root",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=4),
            side=ft.BorderSide(color="cyan", width=2),
        ),
        on_click = lambda e : plot_function(e , tab3)
    )

    graph_row = ft.Row(controls=[graph_label, graph_input_1, graph_input_2, graph_button] , key="graph_row")
    

    tab3.controls.append(graph_row)

    submit_button = ft.TextButton(
        text="Answer",
        key="submit_button_root",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=4),
            side=ft.BorderSide(color="blue", width=2),
        ),
        on_click = lambda e : send_to_backend_root(e , tab3)
    )

    tab3.controls.append(submit_button)
    steps_button = ft.TextButton(
        text="Single Step",
        key="single_step_button_table",
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=4),
            side=ft.BorderSide(color="blue", width=2),
        ),
        on_click=lambda e: send_to_backend_root_single_step(e, tab3)
    )
    tab3.controls.append(steps_button)

    return tab3


def main(page : ft.Page):
    page.scroll = ft.ScrollMode.AUTO
    page.title = "System of Linear Equations"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    tabs = ft.Tabs(
        tabs=
        [
            ft.Tab(content=tab1(), text="Matrix Solver"),
            ft.Tab(content=tab2(), text="Alphabetical Coefficients Solver"),
            ft.Tab(content=tab3(), text="Root Finder"),
        ],
    )
    page.add(tabs)
    matrixTab = page.controls[0].tabs[0].content
    initialize_matrix(matrixTab)
    alphaTab = page.controls[0].tabs[1].content
    initialize_matrix(alphaTab)
    update_suboptions(None , matrixTab)

    rootFinderTab = page.controls[0].tabs[2].content
    update_suboptions_root(None, rootFinderTab)

    
ft.app(target=main)
