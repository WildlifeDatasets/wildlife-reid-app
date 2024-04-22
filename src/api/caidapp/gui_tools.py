from matplotlib import pyplot as plt
import matplotlib
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np
import plotly.graph_objects as go
import plotly
import plotly.subplots
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def np_image_to_bytes(image: np.array, format: str = "png") -> bytes:
    """Convert numpy image to bytes."""
    # Convert numpy image to bytes
    image_bytes = plotly.io.to_image(image, format=format)
    return image_bytes


def np_imgage_to_html(image: np.array, format: str = "png") -> str:
    """Convert numpy image to HTML."""
    # Convert numpy image to HTML
    image_html = plotly.io.to_html(image, format=format)
    return image_html

def array_image_to_html(image: np.array, format: str = "png") -> str:
    data = image.astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(data, 'RGB')

    # Save to a bytes buffer
    buffer = BytesIO()
    image.save(buffer, format=format)
    image_png = buffer.getvalue()

    # Base64 encode
    image_b64 = base64.b64encode(image_png).decode('utf-8')

    html_output = f'<img src="data:image/{format};base64,{image_b64}">'
    return html_output


def create_match_image(
        kp0:list, kp1:list,
        query_image:np.array, database_image:np.array,
        query_name:str, database_name:str,
        num_kp:int=10):
    """Create a visualization of the matches between two images."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cmap = matplotlib.colormaps['Set3']
    for i, (_kp0, _kp1) in enumerate(zip(kp0, kp1)):
        color = cmap(i / num_kp)[:3]
        kwargs = {"fill": False, "radius": 15, "color": color, "linewidth": 2}
        patch = Circle(_kp0, **kwargs)
        ax[0].imshow(query_image)
        ax[0].add_patch(patch)
        patch = Circle(_kp1, **kwargs)
        ax[1].imshow(database_image)
        ax[1].add_patch(patch)
        con = ConnectionPatch(
            xyA=_kp0, xyB=_kp1,
            coordsA="data", coordsB="data",
            axesA=ax[0], axesB=ax[1],
            color=color,
            shrinkA=7, shrinkB=7
        )
        fig.add_artist(con)
    ax[0].set_title(query_name)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_title(database_name)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    # plt.show()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = f"<img src=\'data:image/png;base64,{encoded}\'>"
    return html




def create_match_image_plotly(
        kp0: list, kp1: list,
        query_image: np.array, database_image: np.array,
        query_name: str, database_name: str,
        num_kp: int = 10):
    """Create a visualization of the matches between two images using Plotly."""
    # fig = go.Figure()
    fig = plotly.subplots.make_subplots(rows=1, cols=2, subplot_titles=(query_name, database_name))

    # Normalize keypoints for plotly
    # Plotly expects coordinates in the range of the image dimensions
    query_height, query_width = query_image.shape[:2]
    database_height, database_width = database_image.shape[:2]

    # Normalize keypoints' coordinates to [0, 1]
    kp0_normalized = [(kp[0] / query_width, kp[1] / query_height) for kp in kp0]
    kp1_normalized = [(kp[0] / database_width, kp[1] / database_height) for kp in kp1]

    # Add images to subplots
    # for some reason this blocks the visualization of circles
    fig.add_trace(go.Image(z=query_image), row=1, col=1)
    fig.add_trace(go.Image(z=database_image), row=1, col=2)

    # Colors for keypoints and lines
    colors = plotly.colors.qualitative.Set3

    # Add circles for keypoints and lines connecting them
    for i, ((_kp0, _kp1), color) in enumerate(zip(zip(kp0_normalized, kp1_normalized), colors[:num_kp])):
        x0, y0 = _kp0
        x1, y1 = _kp1

        # Adjustments for circle size relative to image
        circle_size = 0.02  # circle size relative to plot
        # circle_size = 0.2  # circle size relative to plot

        # Add circles on keypoints
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=x0 - circle_size, y0=y0 - circle_size, x1=x0 + circle_size, y1=y0 + circle_size,
                      line_color=color,
                      row=1, col=1)

        fig.add_shape(type="circle",
                      xref="x2", yref="y2",
                      x0=x1 - circle_size, y0=y1 - circle_size, x1=x1 + circle_size, y1=y1 + circle_size,
                      line_color=color,
                      row=1, col=2)

        # # Connect keypoints with lines
        # fig.add_trace(go.Scatter(
        #     x=[x0, x1 + 1], y=[y0, y1],  # Adding 1 to x1 because it is in the second column
        #     mode="lines",
        #     line=dict(color=color, width=2),
        #     xaxis="x", yaxis="y",
        #     # row=1, col=1
        # ))

        # Add line across subplots
        fig.add_shape(
            type="line",
            x0=x0, y0=y0, x1=1, y1=y0,  # Line from point on left image to right edge of left subplot
            line=dict(color=color, width=2),
            xref="x", yref="y",
            row=1, col=1)

        fig.add_shape(
            type="line",
            x0=0, y0=y1, x1=x1, y1=y1,  # Line from left edge of right subplot to point on right image
            line=dict(color=color, width=2),
            xref="x2", yref="y2",
            row=1, col=2)

    # Update axes properties
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # # Set titles
    # fig.update_layout(
    #     title_text=f'{query_name} vs. {database_name}',
    #     showlegend=False,
    #     width=1000, height=500,
    #     template="plotly_white",
    #     grid=dict(columns=2, rows=1),
    #     margin=dict(l=10, r=10, t=30, b=10)
    # )


    # fig.write_image("fig1.png")
    fig.write_html("fig1.html")


    # fig.show()
