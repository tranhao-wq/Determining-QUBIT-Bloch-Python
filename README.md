# Determining-QUBIT-Bloch-Python

<img width="1524" height="831" alt="image" src="https://github.com/user-attachments/assets/7620e8dd-2818-4319-9a04-0fc3e6c1a045" />
<img width="743" height="719" alt="image" src="https://github.com/user-attachments/assets/d583c45a-42d8-413f-b932-57e3d9745937" />
<img width="300" height="216" alt="image" src="https://github.com/user-attachments/assets/3e4fa534-f380-48a9-ad7a-36c822bdbd69" />
<img width="276" height="284" alt="image" src="https://github.com/user-attachments/assets/2bdd4163-29a7-4435-8f8a-f8f1e7beeb84" />
<img width="985" height="711" alt="image" src="https://github.com/user-attachments/assets/4d45d0b2-4771-4a69-a856-3384eac5794a" />

# Bloch Sphere Interactive Visualization

This project provides an interactive 3D visualization of a qubit state on the Bloch sphere using Plotly and ipywidgets.

## Features

* Render a unit Bloch sphere (`radius = 1`).
* Compute qubit state vector components (x, y, z) from angles $\theta$ and $\phi$:

  $$
    x = \sin(\theta)\cos(\phi),
    y = \sin(\theta)\sin(\phi),
    z = \cos(\theta)
  $$
* Interactive sliders to adjust $\theta$ (0 → $\pi$) and $\phi$ (0 → 2$\pi$).
* Real-time update of the state vector on the sphere.
* Pure Python implementation, runs in Jupyter Notebook, JupyterLab, or compatible environments.

## Requirements

* Python 3.7+
* numpy
* plotly
* ipywidgets
* nbformat (>=4.2.0)

```bash
pip install numpy plotly ipywidgets nbformat>=4.2
jupyter nbextension enable --py widgetsnbextension
```

## Usage

1. Clone or download this repository.
2. Open `bloch_interactive.ipynb` (or paste the code into a new Jupyter Notebook).
3. Install dependencies (see above).
4. Run all cells. You will see two sliders and an output area:

   * **θ (theta):** polar angle from 0 to $\pi$.
   * **φ (phi):** azimuthal angle from 0 to 2$\pi$.
5. Adjust sliders to see how the qubit state $|\psi\rangle$ moves on the Bloch sphere in real time.

## Example Notebook Code

```python
import numpy as np
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

# Function to build Bloch sphere figure
def create_bloch_sphere(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Create mesh for sphere
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)

    # Plot sphere surface and state vector
    sphere = go.Surface(x=xs, y=ys, z=zs, opacity=0.2, colorscale='Blues', showscale=False)
    vector = go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode='lines+markers',
                          line=dict(color='red', width=6), marker=dict(size=4, color='red'))

    fig = go.Figure(data=[sphere, vector])
    fig.update_layout(
        scene=dict(aspectmode='cube', xaxis=dict(range=[-1,1]), yaxis=dict(range=[-1,1]), zaxis=dict(range=[-1,1])),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

# Sliders
theta_slider = widgets.FloatSlider(value=np.pi/4, min=0, max=np.pi, step=0.01, description='θ:')
phi_slider   = widgets.FloatSlider(value=np.pi/2, min=0, max=2*np.pi, step=0.01, description='φ:')

output = widgets.Output()

def update_plot(change):
    with output:
        clear_output(wait=True)
        fig = create_bloch_sphere(theta_slider.value, phi_slider.value)
        fig.show(renderer='notebook')

theta_slider.observe(update_plot, names='value')
phi_slider.observe(update_plot, names='value')

# Initial display
update_plot(None)
display(widgets.VBox([theta_slider, phi_slider, output]))
```

## Troubleshooting

* **`ValueError: Mime type rendering requires nbformat>=4.2.0 but it is not installed`**

  * Ensure you have `nbformat>=4.2.0` installed and the Jupyter widget extension enabled.
  * Alternatively, use `fig.show(renderer='notebook_connected')` or switch to `plotly.io.renderers.default = 'notebook'`.

## License

MIT © tth
