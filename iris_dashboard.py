import dash
from dash import dcc, html
import plotly.graph_objs as go
import pickle

# Load the results from the classifier
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)

accuracy_results = results['accuracy_results']
conf_matrix = results['conf_matrix']
classes = results['classes']
best_predictions = results['best_predictions']

# Create a Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1('Iris Dataset Model Evaluation'),
    
    html.Div([
        html.H2('Model Accuracy Comparison'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=list(accuracy_results.keys()),
                        y=list(accuracy_results.values()),
                        marker=dict(color='royalblue'),
                    )
                ],
                layout=go.Layout(
                    title='Model Accuracy',
                    yaxis=dict(title='Accuracy', range=[0, 1]),
                    xaxis=dict(title='Models')
                )
            )
        )
    ]),

    html.Div([
        html.H2('Confusion Matrix for Best Model'),
        dcc.Graph(
            figure=go.Figure(
                data=go.Heatmap(
                    z=conf_matrix,
                    x=classes,
                    y=classes,
                    colorscale='Blues',
                    text=conf_matrix,
                    hoverinfo='text',
                    showscale=True
                ),
                layout=go.Layout(
                    title='Confusion Matrix',
                    xaxis=dict(title='Predicted'),
                    yaxis=dict(title='Actual')
                )
            )
        )
    ]),

    html.Div([
        html.H2('Final Iris Classification'),
        html.Pre(
            "Predictions:\n" + "\n".join(classes[best_predictions]),
            style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
