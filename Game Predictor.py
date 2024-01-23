import PySimpleGUI as sg
import pandas as pd
import json
import joblib

file = open('./Data/NBA Team Data.json')
teams = json.load(file)
current_nba_data = pd.read_csv('./Data/Current NBA Game Data.csv', index_col=0)

# loads the saved preprocessing pipeline and ml model
scaler, rr, training_values, target = joblib.load('./Model/Ridge Classifier Model.pkl')


# retrieves and formats recent NBA game data into a pandas data frame
def get_prediction_stats(away_team, home_team):
    away_team_data = current_nba_data.loc[teams[away_team]]
    away_team_data.index = [f'{i}_A' for i in list(away_team_data.index)]
    home_team_data = current_nba_data.loc[teams[home_team]]
    home_team_data.index = [f'{i}_H' for i in list(home_team_data.index)]

    game_prediction_data = pd.concat([away_team_data, home_team_data])
    game_prediction_data['NEXT_HOME/AWAY'] = 0.0
    game_prediction_data['AWAY_TEAM'] = away_team
    game_prediction_data['HOME_TEAM'] = home_team

    return pd.DataFrame(game_prediction_data).T


def make_prediction(away_team, home_team):
    game_prediction_data = get_prediction_stats(away_team, home_team)

    # applies the preprocessing pipeline to current game prediction data and make a prediciton
    scaled_data = pd.DataFrame(scaler.transform(
        game_prediction_data[training_values]), columns=training_values)
    game_prediction_data["PREDICTED_WINNER"] = rr.predict(scaled_data)

    # finds the predicted winning team
    win_team = away_team if game_prediction_data["PREDICTED_WINNER"][0] == 1.0 else home_team
    return win_team


sg.change_look_and_feel('Default1')

# create a list of teams
teams_list = list(teams.keys())

# defines team selection dropdowns
away_team_dropdown = sg.Combo(teams_list, enable_events=True,
                              default_value=teams_list[0], key='AWAY_DROP', pad=(60, 0))
home_team_dropdown = sg.Combo(teams_list, enable_events=True,
                              default_value=teams_list[1], key='HOME_DROP', pad=(85, 0))
win_name = sg.Text('   ', pad=((65,0), 0), font=("Arial", 13))

# defines team images
FONT = ("Arial", 15, 'bold')
away_img = sg.Image(f'Data/Team Images/ATL.png', key='AWAY_IMG')
home_img = sg.Image(f'Data/Team Images/BOS.png', key='HOME_IMG')
win_img = sg.Image(f'Data/Team Images/Question Mark.png', key='WIN_IMG')

# defines the overall layout of the GUI
layout = [[sg.Text('AWAY TEAM', font=FONT, pad=(20, 0)), sg.Text('HOME TEAM', font=FONT, pad=(50, 0)), sg.Text('WINNER', font=FONT, pad=(35, 0))],
          [away_img, sg.Text('VS', font=FONT), home_img, sg.Push(), win_img],
          [away_team_dropdown, home_team_dropdown, win_name],
          [sg.Push(), sg.Button('Predict'), sg.Push()]]

window = sg.Window('Window Title', layout)

while True:
    # read in events and values from window
    event, values = window.read()

    # exit the program
    if event == sg.WIN_CLOSED:
        break
    
    # update away team image whenever a different team is selected
    if event == 'AWAY_DROP':
        win_img.update(source=f'Data/Team Images/Question Mark.png')
        away_img.update(source=f'Data/Team Images/{values["AWAY_DROP"]}.png')
        win_name.update(value='   ')

    # update home team image whenever a different team is selected
    if event == 'HOME_DROP':
        win_img.update(source=f'Data/Team Images/Question Mark.png')
        home_img.update(source=f'Data/Team Images/{values["HOME_DROP"]}.png')
        win_name.update(value='   ')

    if event == 'Predict':
        # grab team values from dropdown menus
        away_team = values['AWAY_DROP']
        home_team = values['HOME_DROP']
        
        # make a prediction
        pred_win = make_prediction(away_team, home_team)

        # display the predicted team winner
        win_img.update(source=f'Data/Team Images/{pred_win}.png')
        win_name.update(value=pred_win)
        home_img.update(source=f'Data/Team Images/{values["HOME_DROP"]}.png')
        away_img.update(source=f'Data/Team Images/{values["AWAY_DROP"]}.png')


# close the window when done
window.close()
