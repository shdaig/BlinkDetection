import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import plotly.graph_objects as go

paths = [
    'i1.jpg',
    'i2.jpg',
    'i3.jpg']

x = [0, 1, 2]
y = [0, 1, 2]


fig = go.Figure()
for i in range(len(paths)):
    fig.add_layout_image(
            source=Image.open(paths[i]).resize((100, 100)),
            xanchor="center",
            yanchor="middle",
            x=x[i],
            y=y[i],
            xref="x",
            yref="y",
            sizex=1,
            sizey=1,
            opacity=1.0,
            layer="above"
    )
fig.show()