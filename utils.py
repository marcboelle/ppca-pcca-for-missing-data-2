import plotly.graph_objects as go
import numpy as np

from typing import Optional, List

###########################
# Statistical utils functions
###########################

def get_likelihood(X, W, sigma2):
    n, d = X.shape
    C = W @ W.T + sigma2 * np.eye(d)
    X_centered = X-np.mean(X, axis=0)
    S = np.cov(X_centered.T)
    L = -n/2 * (d*np.log(2*np.pi) + np.log(np.linalg.det(C)) + np.trace(np.linalg.inv(C) @ S))
    return L


def introduce_missing_values(X, missing_ratio):
    """
    Randomly introduces missing values into a dataset.
    """
    X_missing = X.copy()
    n_samples, n_features = X.shape
    n_missing = int(missing_ratio * n_samples * n_features)
    missing_indices = np.random.choice(n_samples * n_features, n_missing, replace=False)
    X_missing.flat[missing_indices] = np.nan
    return X_missing


###########################
# Visualization functions
###########################

def plot_fig_projections(X: np.ndarray, projections_2d: np.ndarray, line_scale: float = 1, save_file: Optional[str] = None) -> None:
    """
    Plot the original data and projections on a 2D plane, with lines indicating the projections.
    Only works for 2D data and 1D projections.

    Parameters:
        X: np.ndarray of shape (n_samples, 2)
            The original 2D data to plot.
        projections_2d: np.ndarray of shape (n_samples, 2)
            The projections of the data points in the 2D space.
        line_scale: float, optional, default=1
            The scaling factor for the line representing the principal direction.
        save_file: Optional[str], optional
            If provided, the plot will be saved as a PDF at the specified path.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(size=12, color='blue', line=dict(width=1, color='black')),
        name="Original data",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=projections_2d[:, 0],
        y=projections_2d[:, 1],
        mode='markers',
        marker=dict(size=8, color='darkorange'),
        name="Projections",
        showlegend=False
    ))

    min_projection = projections_2d.min(axis=0)
    max_projection = projections_2d.max(axis=0)

    slope = (max_projection[1] - min_projection[1]) / (max_projection[0] - min_projection[0])

    fig.add_trace(go.Scatter(
        x=[min_projection[0] - line_scale, max_projection[0] + line_scale],
        y=[min_projection[1] - line_scale * slope, max_projection[1] + line_scale * slope],
        mode='lines',
        line=dict(color='darkorange', width=2),
        name="Principal direction",
        showlegend=False
    ))

    for point, projection in zip(X, projections_2d):
        fig.add_trace(go.Scatter(
            x=[point[0], projection[0]],
            y=[point[1], projection[1]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_white",
        plot_bgcolor="white",
        xaxis=dict(scaleanchor="y")
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )

    if save_file:
        fig.write_image(save_file, format='pdf')
        
    fig.show()


def plot_pca_2D(X: np.ndarray, y: np.ndarray, target_names: List[str], colors: List[str], save_path: Optional[str] = None) -> None:
    """
    Plots a 2D PCA scatter plot.

    Parameters:
        X: np.ndarray of shape (n_samples, 2)
            The 2D transformed data (first two principal components).
        y: np.ndarray of shape (n_samples,)
            Array of target labels (e.g., class labels).
        target_names: List[str]
            List of class names corresponding to each label.
        colors: List[str]
            List of colors corresponding to each class label.
        save_path: Optional[str]
            If provided, the plot will be saved to the specified path.
    """

    fig = go.Figure()

    for label, color, name in zip(range(3), colors, target_names):
        fig.add_trace(go.Scatter(
            x=X[y == label, 0],
            y=X[y == label, 1],
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.8),
            name=name
        ))

    fig.update_layout(
        width=500,
        height=500,
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        xaxis=dict(range=[-4, 4]),
        template="plotly_white",
        legend=dict(
            font=dict(size=17),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center", 
            x=0.5
        ),
        font=dict(
            size=15
        )
    )

    fig.show()

    if save_path: 
        fig.write_image(save_path, scale=3)



def plot_CCA_iris(X1_c: np.ndarray, X2_c: np.ndarray, colors: List[str], y: np.ndarray, target_names: List[str], save_path: Optional[str] = None, save : bool = False) -> None:
    """
    Plot the Canonical Correlation Analysis (CCA) projections of two datasets.

    Parameters:
        X1_c: np.ndarray of shape (n_samples, n_features)
            The first dataset after CCA transformation.
        X2_c: np.ndarray of shape (n_samples, n_features)
            The second dataset after CCA transformation.
        colors: List[str]
            List of colors corresponding to each class label.
        y: np.ndarray of shape (n_samples,)
            Array of class labels.
        target_names: List[str]
            List of target class names corresponding to each label.
        save_path: Optional[str]
            If provided, the plot will be saved at the specified path.
    """

    fig = go.Figure()

    for label, color, name in zip(range(3), colors, target_names):
        fig.add_trace(go.Scatter(
            x=X1_c[y == label, 0],
            y=X2_c[y == label, 0],
            mode='markers',
            marker=dict(color=color, size=8, opacity=0.8),
            name=name
        ))

    fig.update_layout(
        xaxis=dict(title=r"Projection of XA"),
        yaxis=dict(title=r"Projection of XB"),
        template="plotly_white",
        width=500,
        height=500,
        font=dict(size=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    fig.show()

    if save:
        fig.write_image(save_path, scale=3)


