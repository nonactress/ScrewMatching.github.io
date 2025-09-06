
from flask import Flask, render_template, request
from badminton_scheduler import schedule, players_master

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    schedule_result = None
    if request.method == 'POST':
        present_players = request.form.getlist('players')
        schedule_result = schedule(players_master, set(present_players))
    
    # Pass the master list of players to the template
    player_names = [player[0] for player in players_master]
    
    return render_template('index.html', players=player_names, schedule_result=schedule_result)

if __name__ == '__main__':
    app.run(debug=True)
