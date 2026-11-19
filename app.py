
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from quantum_utils import (
    normalize_state,
    calculate_probabilities,
    dirac_notation,
    reduced_density_matrix,
    bloch_coordinates,
    entanglement_entropy
)

st.set_page_config(
    page_title="Quantum State Visualiser",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Multi-Qubit State Visualiser")
st.markdown(
    "Explore quantum states, amplitudes, probabilities "
    "and entanglement interactively."
)

st.sidebar.header("Quantum State Configuration")

num_qubits = st.sidebar.selectbox(
    "Select number of qubits",
    [1, 2, 3]
)

dimension = 2 ** num_qubits

defaults = {
    1: "0.707,0.707",
    2: "0.707,0,0,0.707",
    3: "0.707,0,0,0,0,0,0,0.707"
}

st.sidebar.write(f"State vector size: {dimension}")

real_input = st.sidebar.text_input(
    "Real amplitudes",
    value=defaults[num_qubits]
)

imag_input = st.sidebar.text_input(
    "Imaginary amplitudes",
    value=",".join(["0"] * dimension)
)

if st.sidebar.button("Visualise Quantum State"):
    try:
        real = np.array(
            [float(x.strip()) for x in real_input.split(",")]
        )
        imag = np.array(
            [float(x.strip()) for x in imag_input.split(",")]
        )

        if len(real) != dimension or len(imag) != dimension:
            raise ValueError(
                f"Please enter exactly {dimension} values."
            )

        if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
            raise ValueError("Enter finite numeric values only.")

        state = normalize_state(real + 1j * imag)

        st.session_state.quantum_state = state
        st.session_state.num_qubits = num_qubits

    except ValueError as error:
        st.sidebar.error(str(error))

if "quantum_state" not in st.session_state:
    st.info("Enter a quantum state and click Visualise Quantum State.")
    st.stop()

state = st.session_state.quantum_state
n = st.session_state.num_qubits

probabilities = calculate_probabilities(state)

st.success("Quantum state loaded successfully!")

st.subheader("1. Dirac Notation")
st.code(dirac_notation(state))

st.subheader("2. Probability Distribution")

basis_states = [
    format(i, f"0{n}b")
    for i in range(2 ** n)
]

fig = go.Figure(
    go.Bar(
        x=basis_states,
        y=probabilities,
        text=[f"{p:.3f}" for p in probabilities],
        textposition="auto"
    )
)

fig.update_layout(
    title="Measurement Probabilities",
    xaxis_title="Basis States",
    yaxis_title="Probability",
    yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("3. Quantum Amplitudes")

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=basis_states,
    y=state.real,
    name="Real"
))

fig2.add_trace(go.Bar(
    x=basis_states,
    y=state.imag,
    name="Imaginary"
))

fig2.update_layout(
    title="Real and Imaginary Amplitudes",
    barmode="group"
)

st.plotly_chart(fig2, use_container_width=True)


def create_bloch_sphere(x, y, z, qubit):

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)

    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=xs,
        y=ys,
        z=zs,
        opacity=0.2,
        showscale=False,
        colorscale="Blues"
    ))

    fig.add_trace(go.Scatter3d(
        x=[0, x],
        y=[0, y],
        z=[0, z],
        mode="lines+markers",
        line=dict(color="red", width=7),
        marker=dict(size=5),
        name="Qubit State"
    ))

    fig.update_layout(
        title=f"Qubit {qubit + 1} Bloch Sphere",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        height=400
    )

    return fig


st.subheader("4. Bloch Sphere Visualisation")

columns = st.columns(n)

for q in range(n):
    rho = reduced_density_matrix(state, q, n)
    x, y, z = bloch_coordinates(rho)

    with columns[q]:
        st.plotly_chart(
            create_bloch_sphere(x, y, z, q),
            use_container_width=True
        )

        st.write(f"X: {x:.3f}")
        st.write(f"Y: {y:.3f}")
        st.write(f"Z: {z:.3f}")


st.subheader("5. Entanglement Analysis")

for q in range(n):
    rho = reduced_density_matrix(state, q, n)
    entropy = entanglement_entropy(rho)

    st.metric(
        f"Qubit {q + 1} Entanglement Entropy",
        f"{entropy:.4f} bits"
    )

    if entropy < 1e-6:
        st.write("Approximately unentangled with the remaining qubits.")
    else:
        st.write("Entangled with the remaining subsystem.")


st.subheader("6. Quantum Information")

c1, c2, c3 = st.columns(3)

c1.metric("Number of Qubits", n)
c2.metric("State Dimensions", dimension)
c3.metric("Total Probability", f"{sum(probabilities):.4f}")


st.divider()

st.subheader("7. Quantum Concepts Quiz")

questions = [
    (
        "What is quantum superposition?",
        [
            "Combination of quantum basis states",
            "A classical computer",
            "A memory storage technique"
        ],
        0
    ),
    (
        "What is the total probability of a normalized state?",
        ["0", "1", "2"],
        1
    ),
    (
        "What is quantum entanglement?",
        [
            "Quantum correlations between subsystems",
            "A programming language",
            "A type of classical bit"
        ],
        0
    ),
    (
        "What does the Bloch sphere represent?",
        [
            "A single qubit state",
            "A computer network",
            "A database"
        ],
        0
    ),
    (
        "What does squared amplitude magnitude represent?",
        [
            "Measurement probability",
            "Number of qubits",
            "Circuit depth"
        ],
        0
    )
]

answers = []

for i, (question, options, correct) in enumerate(questions):
    selected = st.radio(
        f"Q{i + 1}. {question}",
        options,
        key=f"question_{i}"
    )
    answers.append(options.index(selected) == correct)

if st.button("Submit Quiz"):
    score = sum(answers)
    st.success(f"Your score: {score}/5")

st.divider()

st.caption(
    "Multi-Qubit State Visualiser | Quantum Computing Project | 25AM3504"
)
