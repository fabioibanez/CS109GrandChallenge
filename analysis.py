import pandas as pd
import matplotlib.pyplot as plt
import scipy as stats
import numpy as np
import math

def get_expected_game_per_player(df, average_games_per_year):
    # create a new column for expected games per player
    df['expected_games_per_player'] = average_games_per_year * df['expected_years_left']
    return df


def get_top_10_players_points_naive(years_left):
    # get only the first 10 columns
    df_player_stats = pd.read_csv('player_stats.csv')
    df_player_stats = df_player_stats.head(10)
    
    # adds the number of years left in the NBA for each player
    df_player_stats['expected_years_left'] = years_left

    # get the expected number of games per year
    df_player_stats = get_expected_game_per_player(df_player_stats, 82)

    # linearity of expected points per player
    df_player_stats['expected_points_per_player'] = df_player_stats['expected_games_per_player'] * df_player_stats['PTS']

    df_player_bios = pd.read_csv('player_bios.csv')
    # join on Player and get the age
    df_player_stats = df_player_stats.merge(df_player_bios[['Player', 'Age']], on='Player', how='left')

    # print out the expected_points_per_player column, the years_played column, and the expected_years_left column
    print(df_player_stats[['Player', 'Age','expected_points_per_player', 'expected_years_left']])
    

age_dict = {}
################# MODELING THE EXPECTED NUMBER OF YEARS BASED ON AGE #####################
def get_normal_nba_lifetime_data(retired_dateframe):
    df_retired_players = pd.read_csv('retired_players.csv')
    # using the retired players dataframe get the average number of years player for an age. Use the birth year and the start year to get the start age.
    
    # get rid of players still playing from df, which is characterized by the To column being 2023 or 2022
    df_retired_players = df_retired_players[df_retired_players['To'] != 2023]
    df_retired_players = df_retired_players[df_retired_players['To'] != 2022]

    for index, row in df_retired_players.iterrows():
        # make sure to parse birth year only which is after the comma
        if (bool(str(row['Birth Date']).split(',')) == True and len(str(row['Birth Date']).split(',')) > 1):
            birth_year = str(row['Birth Date']).split(',')
            df_retired_players.at[index, 'start_age'] = int(row['From']) - int(birth_year[1])

    # get the average number of years played for each age
    for index, row in df_retired_players.iterrows():
        df_retired_players.at[index, 'years_played'] = int(row['To']) - int(row['From'])
    
    # make a normal for each age
    for index, row in df_retired_players.iterrows():
        # check if it doesn't already have a list
        if (row['start_age'] not in age_dict):
            age_dict[row['start_age']] = []
        if row['years_played'] > 0:
            age_dict[row['start_age']].append(row['years_played'])
    return df_retired_players

def get_normal_nba_lifetime_plot(age, age_dict):
    # get the mean and standard deviation of the years left
    mean = np.mean(age_dict[age])
    std = np.std(age_dict[age])
    print(mean)
    print(std)
    # get the normal distribution of the years left
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    # Compute endpoints of shaded regions
    left_endpoint = stats.stats.norm.ppf(0.159, loc=mean, scale=std)
    right_endpoint = stats.stats.norm.ppf(0.841, loc=mean, scale=std)
    
    pdf = stats.stats.norm.pdf(x, loc=mean, scale=std)
    # Create plot
    plt.plot(x, pdf, color='black')
    plt.fill_between(x[x >= left_endpoint], pdf[x >= left_endpoint], color='grey', alpha=0.3)
    plt.fill_between(x[x <= right_endpoint], pdf[x <= right_endpoint], color='grey', alpha=0.3)
    plt.plot(x, stats.stats.norm.pdf(x, mean, std))
    plt.title('Normal Distribution of Years Left in NBA for Age ' + str(age))
    plt.show()
    return mean

def get_prob_of_playing_certain_years(age, age_dict, desired_years):
    # use the cdf of the normal to find the probability of playing a certain number of years, specifical the interval age -1 and age + 1
    mean = np.mean(age_dict[age])
    std = np.std(age_dict[age])

    # Generate a normal distribution with mean 0 and standard deviation 1
    x = np.linspace(mean - 3*std, mean + 3*std, 100)
    y = stats.stats.norm.pdf(x, mean, std)

    prob = stats.stats.norm.cdf(desired_years + 1, mean, std) - stats.stats.norm.cdf(desired_years - 1, mean, std)

    # Plot the normal distribution and the probabilities
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.axvline(desired_years, color='r', linestyle='--')
    ax.fill_between(x[x <= desired_years + 1], y[x <= desired_years + 1], color='r', alpha=0.2)
    ax.fill_between(x[x >= desired_years - 1], y[x >= desired_years - 1], color='r', alpha=0.2)
    # hard coded for Luka's years!
    ax.text(-2, 0.1, f'P((X <= 11) and P(X >= 9)) = {prob:.3f}')
    plt.show()

    print(prob)

def main():
    # pass in the number of expected years 
    get_top_10_players_points_naive(9)

    # this function actually creates the age_dict
    get_normal_nba_lifetime_data('retired_players.csv')

    # the first parameter is the age, the second parameter is the age_dict, and the third parameter is the desired number of years
    get_prob_of_playing_certain_years(24, age_dict, 10)

main()