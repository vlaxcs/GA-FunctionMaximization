import csv
import io
import random
import re
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from genetic import GeneticAlgorithm


def evaluate_polynomial(coefficients: list[float], x_value: float) -> float:
    result = coefficients[0]
    for coefficient in coefficients[1:]:
        result = result * x_value + coefficient
    return result


def format_polynomial(coefficients: list[float]) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []

    for index, coefficient in enumerate(coefficients):
        power = degree - index
        if abs(coefficient) < 1e-12:
            continue

        sign = "-" if coefficient < 0 else "+"
        absolute = abs(coefficient)

        if power == 0:
            term_body = f"{absolute:g}"
        elif power == 1:
            if abs(absolute - 1.0) < 1e-12:
                term_body = "x"
            else:
                term_body = f"{absolute:g}x"
        else:
            if abs(absolute - 1.0) < 1e-12:
                term_body = f"x^{power}"
            else:
                term_body = f"{absolute:g}x^{power}"

        if not terms:
            terms.append(term_body if sign == "+" else f"-{term_body}")
        else:
            terms.append(f" {sign} {term_body}")

    return "".join(terms) if terms else "0"


def read_evolution_file(file_path: str = 'Evolutie.txt') -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'File `{file_path}` was not found.')
    return path.read_text(encoding='utf-8')


def parse_first_generation(evolution_text: str) -> list[dict]:
    rows = []
    in_initial_population = False
    pattern = re.compile(r"^\s*(\d+):\s*([01]+)\s*x\s*=\s*([-+0-9.eE]+)\s*f\s*=\s*([-+0-9.eE]+)")

    for line in evolution_text.splitlines():
        stripped = line.strip()

        if stripped.startswith('Populatia initiala'):
            in_initial_population = True
            continue

        if in_initial_population and stripped.startswith('Probabilitati selectie'):
            break

        if not in_initial_population or not stripped:
            continue

        match = pattern.search(line)
        if match:
            index, chromosome, x_value, fitness_value = match.groups()
            rows.append(
                {
                    'index': int(index),
                    'chromosome': chromosome,
                    'x': float(x_value),
                    'f(x)': float(fitness_value),
                }
            )

    if not rows:
        raise ValueError('Could not parse first generation from Evolutie.txt.')

    return rows


def summarize_first_generation(rows: list[dict]) -> dict:
    fitness_values = [row['f(x)'] for row in rows]
    x_values = [row['x'] for row in rows]
    best_row = max(rows, key=lambda row: row['f(x)'])
    worst_row = min(rows, key=lambda row: row['f(x)'])

    return {
        'size': len(rows),
        'mean_fitness': sum(fitness_values) / len(fitness_values),
        'best_fitness': best_row['f(x)'],
        'best_index': best_row['index'],
        'best_chromosome': best_row['chromosome'],
        'best_x': best_row['x'],
        'worst_fitness': worst_row['f(x)'],
        'worst_index': worst_row['index'],
        'x_min': min(x_values),
        'x_max': max(x_values),
    }


def generation_rows_to_csv(rows: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['index', 'chromosome', 'x', 'f(x)'])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


st.set_page_config(page_title='Genetic Function Maximization', layout='wide')
st.title('Function Maximization')
st.caption('University project for Genetic Algorithms')

with st.sidebar:
    st.header('Parameters')
    default_coefficients_by_power = {2: -1.0, 1: 1.0, 0: 2.0}
    custom_degree = st.slider('Polynomial degree', min_value=1, max_value=12, value=2)
    coefficients = []
    for power in range(custom_degree, -1, -1):
        default_value = default_coefficients_by_power.get(power, 0.0)
        coefficient = st.number_input(
            f'Coefficient for x^{power}',
            value=default_value,
            step=0.1,
            key=f'coef_{power}',
        )
        coefficients.append(coefficient)

    lower_bound = st.number_input('Lower bound', value=-1.0)
    upper_bound = st.number_input('Upper bound', value=2.0)
    precision = st.slider('Precision', min_value=0, max_value=8, value=6)

    population_size = st.slider('Population size', min_value=10, max_value=500, value=20)
    generation_count = st.slider('Generations', min_value=5, max_value=500, value=50)
    crossover_probability = st.slider('Crossover probability', min_value=0.0, max_value=1.0, value=0.25)
    mutation_probability = st.slider('Mutation probability', min_value=0.0, max_value=0.2, value=0.01)

    run = st.button('Run Genetic Algorithm', type='primary', width='stretch')

st.subheader('Selected polynomial')
st.code(f"f(x) = {format_polynomial(coefficients)}")

if upper_bound <= lower_bound:
    st.error('Upper bound must be greater than lower bound to visualize and run optimization.')
else:
    sample_count = 250
    step = (upper_bound - lower_bound) / (sample_count - 1)
    x_points = [lower_bound + idx * step for idx in range(sample_count)]
    y_points = [evaluate_polynomial(coefficients, x_value) for x_value in x_points]

    function_figure = go.Figure()
    function_figure.add_trace(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode='lines',
            name='f(x)',
            line={'width': 3},
        )
    )
    function_figure.update_layout(
        title='Function preview',
        xaxis_title='x',
        yaxis_title='f(x)',
        template='plotly_white',
        margin={'l': 20, 'r': 20, 't': 50, 'b': 20},
    )
    st.plotly_chart(function_figure, width='stretch')

    if run:
        run_seed = time.time_ns()
        random.seed(run_seed)

        ga = GeneticAlgorithm(
            population_size=population_size,
            crossover_probability=crossover_probability,
            mutation_probability=mutation_probability,
            generation_count=generation_count,
            generation_rate=1,
            steps=1,
            precision=precision,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            coefficients=coefficients,
        )

        max_history, mean_history = ga.run()
        st.session_state['ga_result'] = {
            'max_history': max_history,
            'mean_history': mean_history,
            'seed': run_seed,
        }

if 'ga_result' in st.session_state:
    max_history = st.session_state['ga_result']['max_history']
    mean_history = st.session_state['ga_result']['mean_history']
    generations = list(range(1, len(max_history) + 1))

    best_value = max(max_history)
    best_generation = max_history.index(best_value) + 1

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric('Best fitness', f'{best_value:.6f}')
    metric_col_2.metric('Best generation', best_generation)
    metric_col_3.metric('Final mean fitness', f'{mean_history[-1]:.6f}')
    st.caption(f"Seed used: {st.session_state['ga_result']['seed']}")

    evolution_figure = go.Figure()
    evolution_figure.add_trace(
        go.Scatter(
            x=generations,
            y=max_history,
            mode='lines',
            name='Max fitness',
        )
    )
    evolution_figure.add_trace(
        go.Scatter(
            x=generations,
            y=mean_history,
            mode='lines',
            name='Mean fitness',
        )
    )
    evolution_figure.update_layout(
        title='Fitness evolution',
        xaxis_title='Generation',
        yaxis_title='Fitness',
        template='plotly_white',
    )
    st.plotly_chart(evolution_figure, width='stretch')

    controls_col_1, controls_col_2 = st.columns(2)
    analyze_first_generation = controls_col_1.button('Analyze first generation', width='stretch')
    evolution_txt = Path('Evolutie.txt')
    if evolution_txt.exists():
        controls_col_2.download_button(
            label='Export Evolutie.txt',
            data=evolution_txt.read_text(encoding='utf-8'),
            file_name='Evolutie.txt',
            mime='text/plain',
            width='stretch',
        )
    else:
        controls_col_2.button('Export Evolutie.txt', width='stretch', disabled=True)

    if analyze_first_generation:
        try:
            evolution_text = read_evolution_file()
            first_generation_rows = parse_first_generation(evolution_text)
            summary = summarize_first_generation(first_generation_rows)
            st.session_state['first_generation_analysis'] = {
                'rows': first_generation_rows,
                'summary': summary,
            }
        except (FileNotFoundError, ValueError) as error:
            st.session_state.pop('first_generation_analysis', None)
            st.error(str(error))

    if 'first_generation_analysis' in st.session_state:
        analysis = st.session_state['first_generation_analysis']
        summary = analysis['summary']
        rows = analysis['rows']

        st.subheader('First generation analysis')
        metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
        metric_col_1.metric('Population', summary['size'])
        metric_col_2.metric('Mean f(x)', f"{summary['mean_fitness']:.6f}")
        metric_col_3.metric('Best f(x)', f"{summary['best_fitness']:.6f}")
        metric_col_4.metric('Worst f(x)', f"{summary['worst_fitness']:.6f}")
        st.caption(
            f"Best individual: #{summary['best_index']} | x = {summary['best_x']:.6f} | "
            f"chromosome = {summary['best_chromosome']}"
        )
        st.dataframe(rows, width='stretch', hide_index=True)
        st.download_button(
            label='Export first generation CSV',
            data=generation_rows_to_csv(rows),
            file_name='first_generation.csv',
            mime='text/csv',
            width='stretch',
        )
else:
    st.info('Set parameters from the sidebar and click `Run Genetic Algorithm`.')
