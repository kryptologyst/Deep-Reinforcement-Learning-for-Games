"""
Interactive Web UI for Deep Reinforcement Learning Training Monitoring
This dashboard provides real-time visualization of training progress and model evaluation.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Any

# Import our DQN components
from 0140 import AdvancedDQNAgent, TrainingDatabase, train_dqn_agent, evaluate_agent

class TrainingDashboard:
    def __init__(self, db_path: str = "training_results.db"):
        self.db_path = db_path
        self.db = TrainingDatabase(db_path)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
        # Training state
        self.training_thread = None
        self.is_training = False
        self.current_session_id = None
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🎮 Deep Reinforcement Learning Training Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎛️ Training Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Episodes:"),
                                    dbc.Input(id="episodes-input", type="number", value=500, min=1, max=10000)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Learning Rate:"),
                                    dbc.Input(id="lr-input", type="number", value=0.001, step=0.0001, min=0.0001)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Batch Size:"),
                                    dbc.Input(id="batch-size-input", type="number", value=64, min=1, max=512)
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Gamma:"),
                                    dbc.Input(id="gamma-input", type="number", value=0.99, step=0.01, min=0.1, max=1.0)
                                ], width=3)
                            ], className="mb-3"),
                            
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🚀 Start Training", id="start-training-btn", 
                                              color="success", className="me-2"),
                                    dbc.Button("⏹️ Stop Training", id="stop-training-btn", 
                                              color="danger", className="me-2"),
                                    dbc.Button("🎯 Evaluate Model", id="evaluate-btn", 
                                              color="info", className="me-2"),
                                    dbc.Button("📊 Refresh Data", id="refresh-btn", 
                                              color="secondary")
                                ])
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Status and Alerts
            dbc.Row([
                dbc.Col([
                    dbc.Alert(id="status-alert", is_open=False, duration=4000)
                ])
            ]),
            
            # Real-time Training Progress
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Real-time Training Progress"),
                        dbc.CardBody([
                            dcc.Graph(id="training-progress-graph"),
                            dcc.Interval(id="progress-interval", interval=2000, n_intervals=0)
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Training Metrics Dashboard
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Episode Rewards"),
                        dbc.CardBody([
                            dcc.Graph(id="rewards-graph")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📉 Training Loss"),
                        dbc.CardBody([
                            dcc.Graph(id="loss-graph")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Model Performance and Evaluation
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Model Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-graph")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⚙️ Hyperparameters"),
                        dbc.CardBody([
                            html.Div(id="hyperparams-display")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Training Sessions History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📚 Training Sessions History"),
                        dbc.CardBody([
                            dbc.Table(id="sessions-table", striped=True, bordered=True, hover=True)
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output("status-alert", "is_open"),
             Output("status-alert", "children"),
             Output("status-alert", "color"),
             Output("current-session-id", "data")],
            [Input("start-training-btn", "n_clicks"),
             Input("stop-training-btn", "n_clicks"),
             Input("evaluate-btn", "n_clicks")],
            [State("episodes-input", "value"),
             State("lr-input", "value"),
             State("batch-size-input", "value"),
             State("gamma-input", "value")]
        )
        def handle_training_controls(start_clicks, stop_clicks, eval_clicks, 
                                   episodes, lr, batch_size, gamma):
            ctx = callback_context
            if not ctx.triggered:
                return False, "", "info", None
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-training-btn" and not self.is_training:
                # Start training in a separate thread
                self.start_training(episodes, lr, batch_size, gamma)
                return True, f"🚀 Training started with {episodes} episodes!", "success", self.current_session_id
            
            elif button_id == "stop-training-btn" and self.is_training:
                self.stop_training()
                return True, "⏹️ Training stopped!", "warning", None
            
            elif button_id == "evaluate-btn":
                self.evaluate_model()
                return True, "🎯 Model evaluation completed!", "info", None
            
            return False, "", "info", None
        
        @self.app.callback(
            [Output("training-progress-graph", "figure"),
             Output("rewards-graph", "figure"),
             Output("loss-graph", "figure"),
             Output("performance-graph", "figure"),
             Output("hyperparams-display", "children"),
             Output("sessions-table", "children")],
            [Input("progress-interval", "n_intervals"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_dashboard(n_intervals, refresh_clicks):
            """Update all dashboard components"""
            
            # Get latest training data
            if self.current_session_id:
                session_data = self.db.get_session_results(self.current_session_id)
                if session_data:
                    df = pd.DataFrame(session_data)
                    
                    # Training progress graph
                    progress_fig = go.Figure()
                    progress_fig.add_trace(go.Scatter(
                        x=df['episode'], y=df['reward'],
                        mode='lines+markers', name='Reward',
                        line=dict(color='blue', width=2)
                    ))
                    progress_fig.update_layout(
                        title="Real-time Training Progress",
                        xaxis_title="Episode",
                        yaxis_title="Reward",
                        template="plotly_white"
                    )
                    
                    # Rewards graph
                    rewards_fig = px.line(df, x='episode', y='reward', 
                                        title="Episode Rewards Over Time")
                    rewards_fig.update_layout(template="plotly_white")
                    
                    # Loss graph
                    loss_fig = px.line(df, x='episode', y='loss', 
                                     title="Training Loss Over Time")
                    loss_fig.update_layout(template="plotly_white")
                    
                    # Performance graph
                    performance_fig = go.Figure()
                    performance_fig.add_trace(go.Scatter(
                        x=df['episode'], y=df['steps'],
                        mode='lines', name='Episode Length',
                        line=dict(color='green', width=2)
                    ))
                    performance_fig.update_layout(
                        title="Episode Length (Performance)",
                        xaxis_title="Episode",
                        yaxis_title="Steps",
                        template="plotly_white"
                    )
                    
                    # Hyperparameters display
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT hyperparameters FROM training_sessions 
                        WHERE id = ?
                    ''', (self.current_session_id,))
                    hyperparams = json.loads(cursor.fetchone()[0])
                    conn.close()
                    
                    hyperparams_html = [
                        html.H6("Current Training Parameters:"),
                        html.Ul([
                            html.Li(f"Episodes: {hyperparams.get('episodes', 'N/A')}"),
                            html.Li(f"Learning Rate: {hyperparams.get('lr', 'N/A')}"),
                            html.Li(f"Gamma: {hyperparams.get('gamma', 'N/A')}"),
                            html.Li(f"Batch Size: {hyperparams.get('batch_size', 'N/A')}"),
                            html.Li(f"Epsilon Decay: {hyperparams.get('epsilon_decay', 'N/A')}")
                        ])
                    ]
                    
                else:
                    # Empty graphs if no data
                    progress_fig = go.Figure()
                    rewards_fig = go.Figure()
                    loss_fig = go.Figure()
                    performance_fig = go.Figure()
                    hyperparams_html = [html.P("No training data available")]
            else:
                progress_fig = go.Figure()
                rewards_fig = go.Figure()
                loss_fig = go.Figure()
                performance_fig = go.Figure()
                hyperparams_html = [html.P("No active training session")]
            
            # Sessions table
            sessions_table = self.get_sessions_table()
            
            return (progress_fig, rewards_fig, loss_fig, performance_fig, 
                   hyperparams_html, sessions_table)
    
    def start_training(self, episodes: int, lr: float, batch_size: int, gamma: float):
        """Start training in a separate thread"""
        if self.is_training:
            return
        
        self.is_training = True
        
        def training_worker():
            try:
                # Create a custom agent with specified hyperparameters
                env = gym.make("CartPole-v1")
                state_dim = env.observation_space.shape[0]
                action_dim = env.action_space.n
                
                agent = AdvancedDQNAgent(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    lr=lr,
                    gamma=gamma,
                    batch_size=batch_size
                )
                
                # Create training session
                session_name = f"Dashboard_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                hyperparameters = {
                    'episodes': episodes,
                    'lr': lr,
                    'gamma': gamma,
                    'batch_size': batch_size,
                    'epsilon_decay': 0.995,
                    'min_epsilon': 0.01,
                    'target_update': 10
                }
                
                self.current_session_id = self.db.create_session(session_name, hyperparameters)
                
                # Training loop
                for episode in range(episodes):
                    if not self.is_training:  # Check for stop signal
                        break
                    
                    state, _ = env.reset()
                    total_reward = 0
                    steps = 0
                    episode_loss = 0
                    loss_count = 0
                    
                    done = False
                    while not done:
                        action = agent.select_action(state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        agent.store_transition(state, action, reward, next_state, done)
                        
                        loss = agent.train()
                        if loss > 0:
                            episode_loss += loss
                            loss_count += 1
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                    
                    # Log episode results
                    avg_loss = episode_loss / max(loss_count, 1)
                    self.db.log_episode(self.current_session_id, episode, total_reward, 
                                      agent.epsilon, avg_loss, steps)
                    
                    # Update agent
                    agent.decay_epsilon()
                    if episode % agent.target_update == 0:
                        agent.update_target_network()
                    
                    # Small delay to allow UI updates
                    time.sleep(0.1)
                
                env.close()
                
            except Exception as e:
                print(f"Training error: {e}")
            finally:
                self.is_training = False
        
        self.training_thread = threading.Thread(target=training_worker)
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training(self):
        """Stop the current training"""
        self.is_training = False
        if self.training_thread:
            self.training_thread.join(timeout=5)
    
    def evaluate_model(self):
        """Evaluate the current model"""
        if not self.current_session_id:
            return
        
        # This would load the latest model and evaluate it
        # Implementation depends on model saving/loading logic
        pass
    
    def get_sessions_table(self):
        """Get training sessions table data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_name, start_time, end_time, total_episodes, final_score
            FROM training_sessions
            ORDER BY start_time DESC
            LIMIT 10
        ''')
        
        sessions = cursor.fetchall()
        conn.close()
        
        if not sessions:
            return dbc.Table.from_dataframe(
                pd.DataFrame(columns=["Session", "Start Time", "End Time", "Episodes", "Score"]),
                striped=True, bordered=True, hover=True
            )
        
        df = pd.DataFrame(sessions, columns=["Session", "Start Time", "End Time", "Episodes", "Score"])
        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    
    def run(self, debug: bool = True, port: int = 8050):
        """Run the dashboard"""
        print(f"🚀 Starting DRL Training Dashboard on http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

# Additional utility functions for the dashboard
def create_model_comparison_chart(session_ids: List[int], db_path: str = "training_results.db"):
    """Create a comparison chart for multiple training sessions"""
    conn = sqlite3.connect(db_path)
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set1
    
    for i, session_id in enumerate(session_ids):
        cursor = conn.cursor()
        cursor.execute('''
            SELECT episode, reward FROM episode_results 
            WHERE session_id = ? ORDER BY episode
        ''', (session_id,))
        
        data = cursor.fetchall()
        if data:
            episodes, rewards = zip(*data)
            fig.add_trace(go.Scatter(
                x=episodes, y=rewards,
                mode='lines',
                name=f'Session {session_id}',
                line=dict(color=colors[i % len(colors)])
            ))
    
    conn.close()
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Episode",
        yaxis_title="Reward",
        template="plotly_white"
    )
    
    return fig

def export_training_data(session_id: int, format: str = "csv", db_path: str = "training_results.db"):
    """Export training data in various formats"""
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    cursor.execute('''
        SELECT episode, reward, epsilon, loss, steps
        FROM episode_results
        WHERE session_id = ?
        ORDER BY episode
    ''', (session_id,))
    
    data = cursor.fetchall()
    conn.close()
    
    df = pd.DataFrame(data, columns=["episode", "reward", "epsilon", "loss", "steps"])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format.lower() == "csv":
        filename = f"training_data_session_{session_id}_{timestamp}.csv"
        df.to_csv(filename, index=False)
    elif format.lower() == "json":
        filename = f"training_data_session_{session_id}_{timestamp}.json"
        df.to_json(filename, orient="records", indent=2)
    elif format.lower() == "excel":
        filename = f"training_data_session_{session_id}_{timestamp}.xlsx"
        df.to_excel(filename, index=False)
    
    return filename

if __name__ == "__main__":
    # Create and run the dashboard
    dashboard = TrainingDashboard()
    dashboard.run(debug=True, port=8050)
