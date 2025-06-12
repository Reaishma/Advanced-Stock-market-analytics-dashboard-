import plotly.express as px
import plotly.subplots as sp
import pandas as pd

# Sample dataset for top batsmen
batsmen_data = {
    'Player': ['Virat Kohli', 'Rohit Sharma', 'Suryakumar Yadav', 'KL Rahul', 'Shubman Gill'],
    'Runs': [700, 650, 600, 550, 500],
    'Strike Rate': [130, 120, 140, 110, 125]
}

# Sample dataset for top bowlers
bowlers_data = {
    'Player': ['Jasprit Bumrah', 'Mohammed Shami', 'Yuzvendra Chahal', 'Ravichandran Ashwin', 'Kuldeep Yadav'],
    'Wickets': [25, 20, 18, 15, 12]
}

# Sample dataset for team standings
team_data = {
    'Team': ['MI', 'CSK', 'RCB', 'DC', 'KKR'],
    'Points': [20, 18, 16, 14, 12]
}

# Create DataFrames
batsmen_df = pd.DataFrame(batsmen_data)
bowlers_df = pd.DataFrame(pd.DataFrame(bowlers_data))
team_df = pd.DataFrame(team_data)

# Create figures
fig_batsmen = px.bar(batsmen_df, x='Player', y='Runs', title='Top Batsmen by Runs')
fig_bowlers = px.bar(bowlers_df, x='Player', y='Wickets', title='Top Bowlers by Wickets')
fig_team = px.bar(team_df, x='Team', y='Points', title='Team Standings')
fig_performance = px.scatter(batsmen_df, x='Runs', y='Strike Rate', hover_name='Player', title='Player Performance')

# Create dashboard
fig_dashboard = sp.make_subplots(rows=2, cols=2, subplot_titles=['Top Batsmen', 'Top Bowlers', 'Team Standings', 'Player Performance'])

fig_dashboard.add_trace(fig_batsmen.data[0], row=1, col=1)
fig_dashboard.add_trace(fig_bowlers.data[0], row=1, col=2)
fig_dashboard.add_trace(fig_team.data[0], row=2, col=1)
fig_dashboard.add_trace(fig_performance.data[0], row=2, col=2)

fig_dashboard.update_layout(height=800, width=1200, title_text="IPL Dashboard")

# Show dashboard
fig_dashboard.show()