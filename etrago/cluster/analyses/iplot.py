

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as off

import pandas as pd

df = pd.read_csv('/tmp/data.csv', index_col=['s','k'])

data = [
    go.Surface(
        z=df['RE-objective'].unstack('k').as_matrix(),
        x=df.index.get_level_values('k').unique(),
        y=df.index.get_level_values('s').unique()
    )
]

layout = go.Layout(
    title='Relative error objective function',
    showlegend=True,
    scene=dict(
        xaxis=dict(title='Regional cluster k'),
        yaxis=dict(title='Temporal cluster s'),
        zaxis=dict(title='Relative deviation in %')
        #camera=dict(
        #    eye=dict(x=2, y=-1.7, z=1)
        #)
    ),
    autosize=False,
    width=1000,
    height=1000,
    margin=dict(
        l=100, r=100, b=100, t=100
    )
)

fig = go.Figure(data=data, layout=layout)
# needs log in
# py.image.save_as(fig, 'plot.pdf') 
off.plot(fig, filename='e-highway-capacities.html')
