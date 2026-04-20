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
    pattern = re.compile(r"^\s*(\d+):\s*([01]+)\s*x\s*=\s*([-+0-9.,eE]+)\s*f\s*=\s*([-+0-9.,eE]+)")

    for line in evolution_text.splitlines():
        stripped = line.strip()

        if stripped.startswith('Populatia initiala') or stripped.startswith('Populatie initiala'):
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
                        'x': parse_float(x_value),
                        'f(x)': parse_float(fitness_value),
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


def parse_float(value: str) -> float:
    return float(value.replace(',', '.'))


def parse_population_section(evolution_text: str, section_titles: list[str]) -> list[dict]:
    lines = evolution_text.splitlines()
    pattern = re.compile(r"^\s*(\d+):\s*([01]+)\s*x\s*=\s*([-+0-9.,eE]+)\s*f\s*=\s*([-+0-9.,eE]+)\s*$")
    rows = []
    in_section = False

    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(title) for title in section_titles):
            in_section = True
            continue

        if not in_section:
            continue

        if not stripped:
            if rows:
                break
            continue

        match = pattern.match(stripped)
        if not match:
            if rows:
                break
            continue

        index, chromosome, x_value, fitness_value = match.groups()
        rows.append(
            {
                'index': int(index),
                'chromosome': chromosome,
                'x': parse_float(x_value),
                'f(x)': parse_float(fitness_value),
            }
        )

    return rows


def parse_selection_details(evolution_text: str) -> dict:
    lines = evolution_text.splitlines()
    probability_pattern = re.compile(r"^\s*(\d+):\s*([01]+)\s+probabilitiate\s*=\s*([-+0-9.,eE]+)\s*$")
    draw_pattern = re.compile(r"^u\s*=\s*([-+0-9.,eE]+)\s+selectam cromozomul\s+(\d+)\s*$")
    probabilities = []
    intervals = []
    draws = []
    in_probabilities = False
    in_intervals = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Probabilitati selectie'):
            in_probabilities = True
            in_intervals = False
            continue

        if stripped.startswith('Intervale probabilitati selectie'):
            in_probabilities = False
            in_intervals = True
            continue

        if stripped.startswith('Dupa selectie'):
            break

        if in_probabilities:
            match = probability_pattern.match(stripped)
            if match:
                index, chromosome, probability = match.groups()
                probabilities.append(
                    {
                        'index': int(index),
                        'chromosome': chromosome,
                        'probability': parse_float(probability),
                    }
                )

        if in_intervals:
            draw_match = draw_pattern.match(stripped)
            if draw_match:
                draw_value, selected_idx = draw_match.groups()
                draws.append({'u': parse_float(draw_value), 'selected_idx': int(selected_idx)})
                continue

            if stripped:
                try:
                    intervals.append(parse_float(stripped))
                except ValueError:
                    pass

    return {
        'probabilities': probabilities,
        'intervals': intervals,
        'draws': draws,
    }


def parse_crossover_details(evolution_text: str) -> dict:
    lines = evolution_text.splitlines()
    parent_pattern = re.compile(r"^\s*(\d+):\s*([01]+)\s*u\s*=\s*([-+0-9.,eE]+)(.*)$")
    event_pattern = re.compile(r"^Recombinare dintre cromozomul\s+(\d+)\s+cu cromozomul\s+(\d+)\s+la punctul\s+(\d+)\s*$")
    before_pattern = re.compile(r"^BEF:\s*([01]+)\s*<->\s*([01]+)\s*$")
    after_pattern = re.compile(r"^AFT:\s*([01]+)\s*<->\s*([01]+)\s*$")
    threshold = None
    parents = []
    events = []
    in_crossover_section = False
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if stripped.startswith('Probabilitatea de incrucisare'):
            in_crossover_section = True
            try:
                threshold = parse_float(stripped.split()[-1])
            except ValueError:
                threshold = None
            idx += 1
            continue

        if in_crossover_section:
            if stripped.startswith('Dupa recombinare'):
                break

            parent_match = parent_pattern.match(stripped)
            if parent_match:
                chromosome_idx, chromosome, u_value, tail = parent_match.groups()
                parents.append(
                    {
                        'index': int(chromosome_idx),
                        'chromosome': chromosome,
                        'u': parse_float(u_value),
                        'participates': 'participa' in tail,
                    }
                )
                idx += 1
                continue

            event_match = event_pattern.match(stripped)
            if event_match and idx + 2 < len(lines):
                p1_idx, p2_idx, point = event_match.groups()
                before_line = lines[idx + 1].strip()
                after_line = lines[idx + 2].strip()
                before_match = before_pattern.match(before_line)
                after_match = after_pattern.match(after_line)
                if before_match and after_match:
                    bef1, bef2 = before_match.groups()
                    aft1, aft2 = after_match.groups()
                    events.append(
                        {
                            'p1_idx': int(p1_idx),
                            'p2_idx': int(p2_idx),
                            'point': int(point),
                            'before_p1': bef1,
                            'before_p2': bef2,
                            'after_p1': aft1,
                            'after_p2': aft2,
                        }
                    )
                idx += 3
                continue

        idx += 1

    return {
        'threshold': threshold,
        'parents': parents,
        'events': events,
    }


def parse_mutation_changes(evolution_text: str) -> list[int]:
    lines = evolution_text.splitlines()
    modified_pattern = re.compile(r"^\s*(\d+)\s*$")
    modified = []
    in_modified_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Au fost modificati cromozomii'):
            in_modified_block = True
            continue

        if in_modified_block:
            if stripped.startswith('Dupa mutatie') or stripped.startswith('Evolutia maximului'):
                break
            match = modified_pattern.match(stripped)
            if match:
                modified.append(int(match.group(1)))

    return modified


def parse_evolution_details(evolution_text: str) -> dict:
    initial_rows = parse_population_section(evolution_text, ['Populatia initiala', 'Populatie initiala'])
    selected_rows = parse_population_section(evolution_text, ['Dupa selectie'])
    recombined_rows = parse_population_section(evolution_text, ['Dupa recombinare'])
    mutated_rows = parse_population_section(evolution_text, ['Dupa mutatie'])
    selection = parse_selection_details(evolution_text)
    crossover = parse_crossover_details(evolution_text)
    mutation_changed = parse_mutation_changes(evolution_text)

    return {
        'initial_rows': initial_rows,
        'selected_rows': selected_rows,
        'recombined_rows': recombined_rows,
        'mutated_rows': mutated_rows,
        'selection': selection,
        'crossover': crossover,
        'mutation_changed': mutation_changed,
    }


def build_first_generation_analysis(file_path: str = 'Evolutie.txt') -> dict:
    evolution_text = read_evolution_file(file_path)
    first_generation_rows = parse_first_generation(evolution_text)
    summary = summarize_first_generation(first_generation_rows)
    details = parse_evolution_details(evolution_text)
    return {
        'rows': first_generation_rows,
        'summary': summary,
        'details': details,
    }


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
st.caption('This is the exact function optimized by the genetic algorithm.')

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
    st.caption('Function preview: shows how f(x) behaves on the selected interval before running GA.')
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
        try:
            st.session_state['first_generation_analysis'] = build_first_generation_analysis()
        except (FileNotFoundError, ValueError):
            st.session_state.pop('first_generation_analysis', None)

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
    st.caption('Fitness evolution: max fitness is the best individual each generation, mean fitness is overall population quality.')
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
            st.session_state['first_generation_analysis'] = build_first_generation_analysis()
        except (FileNotFoundError, ValueError) as error:
            st.session_state.pop('first_generation_analysis', None)
            st.error(str(error))

    if 'first_generation_analysis' in st.session_state:
        analysis = st.session_state['first_generation_analysis']
        summary = analysis['summary']
        rows = analysis['rows']

        st.subheader('First generation analysis')
        st.caption('This section explains what happened specifically in generation 1, using Evolutie.txt logs.')
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

        details = analysis['details']
        selection = details['selection']
        crossover = details['crossover']

        st.subheader('Selection dynamics')
        if selection['probabilities'] and selection['draws']:
            population_indices = [row['index'] for row in selection['probabilities']]
            probability_values = [row['probability'] for row in selection['probabilities']]
            draw_count = len(selection['draws'])
            selected_counts = {index: 0 for index in population_indices}
            for draw in selection['draws']:
                selected_counts[draw['selected_idx']] = selected_counts.get(draw['selected_idx'], 0) + 1
            selected_relative = [
                selected_counts[index] / draw_count if draw_count else 0.0
                for index in population_indices
            ]

            probability_figure = go.Figure()
            probability_figure.add_trace(
                go.Bar(
                    x=population_indices,
                    y=probability_values,
                    name='Theoretical probability',
                )
            )
            probability_figure.add_trace(
                go.Bar(
                    x=population_indices,
                    y=selected_relative,
                    name='Observed frequency (first generation)',
                )
            )
            probability_figure.update_layout(
                barmode='group',
                title='Selection probability vs observed picks',
                xaxis_title='Chromosome index',
                yaxis_title='Value',
                template='plotly_white',
            )
            st.caption(
                'Selection probability vs observed picks: compares expected roulette probability with actual selection frequency.'
            )
            st.plotly_chart(probability_figure, width='stretch')

            roulette_figure = go.Figure()
            if selection['intervals']:
                roulette_figure.add_trace(
                    go.Scatter(
                        x=list(range(len(selection['intervals']))),
                        y=selection['intervals'],
                        mode='lines+markers',
                        name='Cumulative intervals',
                    )
                )
            roulette_figure.add_trace(
                go.Scatter(
                    x=[draw['selected_idx'] for draw in selection['draws']],
                    y=[draw['u'] for draw in selection['draws']],
                    mode='markers',
                    name='Random draws u',
                    marker={
                        'size': 10,
                        'color': [draw['selected_idx'] for draw in selection['draws']],
                        'colorscale': 'Viridis',
                        'showscale': True,
                        'colorbar': {'title': 'Selected idx'},
                    },
                )
            )
            roulette_figure.update_layout(
                title='Roulette wheel map (intervals and selections)',
                xaxis_title='Chromosome index / interval boundary',
                yaxis_title='u in [0, 1]',
                template='plotly_white',
            )
            st.caption(
                'Roulette wheel map: each dot is a random draw u; where it lands on cumulative intervals decides selected chromosome.'
            )
            st.plotly_chart(roulette_figure, width='stretch')
        else:
            st.info('No selection details found in Evolutie.txt.')

        st.subheader('Crossover and mutation')
        if crossover['parents']:
            crossover_figure = go.Figure()
            crossover_figure.add_trace(
                go.Bar(
                    x=[row['index'] for row in crossover['parents']],
                    y=[row['u'] for row in crossover['parents']],
                    marker={
                        'color': ['#16a34a' if row['participates'] else '#9ca3af' for row in crossover['parents']]
                    },
                    name='u value per chromosome',
                )
            )
            if crossover['threshold'] is not None:
                crossover_figure.add_hline(
                    y=crossover['threshold'],
                    line_dash='dash',
                    line_color='#dc2626',
                    annotation_text=f"threshold = {crossover['threshold']}",
                    annotation_position='top left',
                )
            crossover_figure.update_layout(
                title='Crossover participation threshold',
                xaxis_title='Chromosome index',
                yaxis_title='u',
                template='plotly_white',
            )
            st.caption(
                'Crossover participation: bars under the red threshold participate in recombination, others remain unchanged at this step.'
            )
            st.plotly_chart(crossover_figure, width='stretch')
        else:
            st.info('No crossover participation section found in Evolutie.txt.')

        if details['recombined_rows'] and details['mutated_rows']:
            recombined_by_idx = {row['index']: row for row in details['recombined_rows']}
            mutated_by_idx = {row['index']: row for row in details['mutated_rows']}
            common_indices = sorted(set(recombined_by_idx).intersection(mutated_by_idx))
            mutation_delta_rows = []
            for index in common_indices:
                before_row = recombined_by_idx[index]
                after_row = mutated_by_idx[index]
                mutation_delta_rows.append(
                    {
                        'index': index,
                        'delta_f': after_row['f(x)'] - before_row['f(x)'],
                        'changed': before_row['chromosome'] != after_row['chromosome'],
                    }
                )

            mutation_figure = go.Figure()
            mutation_figure.add_trace(
                go.Bar(
                    x=[row['index'] for row in mutation_delta_rows],
                    y=[row['delta_f'] for row in mutation_delta_rows],
                    marker={
                        'color': ['#f97316' if row['changed'] else '#d1d5db' for row in mutation_delta_rows]
                    },
                    name='delta f(x) from mutation',
                )
            )
            mutation_figure.update_layout(
                title='Mutation impact by chromosome (after - before)',
                xaxis_title='Chromosome index',
                yaxis_title='delta f(x)',
                template='plotly_white',
            )
            st.caption(
                'Mutation impact: positive bars mean mutation improved fitness, negative bars mean it reduced fitness.'
            )
            st.plotly_chart(mutation_figure, width='stretch')

            st.caption(
                f"Mutated chromosomes logged: {', '.join(str(idx) for idx in details['mutation_changed']) if details['mutation_changed'] else 'none'}"
            )

        if crossover['events']:
            st.write('Recombination events (first generation)')
            st.caption('Each row shows parent pair, crossover point, and resulting offspring chromosomes.')
            st.dataframe(crossover['events'], width='stretch', hide_index=True)
else:
    st.info('Set parameters from the sidebar and click `Run Genetic Algorithm`.')

