import pandas as pd
import streamlit as st
import numpy as np
import random
from deap import base, creator, tools
import altair as alt
from PIL import Image
from snowflake.snowpark import Session
import time

## -- To generate requirements.txt:  pip list --format=freeze > requirements.txt

# =====  SET UP  =====
st.set_page_config(
    page_title="Squadmateo",
    #initial_sidebar_state="collapsed",
    )

image = Image.open('logo.png')

st.image(image, use_column_width=True)


# =====  SIDEBAR  =====
with st.sidebar:
    st.markdown("## About")
    st.markdown("# Welcome to SQUADMATEO")
    st.write("Our app leverages a sophisticated genetic algorithm to produce two teams of players that are evenly matched in both skill level and total points scored. This algorithm is powered by the amazing data cloud of Snowflake, which provides the list of players and their scores.")
    st.markdown("Created by Diego Garzon")
    st.markdown("for the Snowflake Summit Hackathon 2023")
    
    
# ******************** NEW ***********************    

# Establish Snowflake session
@st.cache_resource
def create_session():
    return Session.builder.configs(st.secrets.snowflake).create()

session = create_session()
#st.success("Connected to Snowflake!")

# Load data table
@st.cache_data
def load_data(table_name):
    ## Read in data table
    # st.write(f"Here's some example data from `{table_name}`:")
    table = session.table(table_name)
    
    ## Do some computation on it
    table = table.limit(100)
    
    ## Collect the results. This will run the query and download the data
    table = table.collect()
    return table

# Select and display data table
table_name = "DB_PLAYERS.PUBLIC.PLAYERS"
sf = load_data(table_name)
sf = pd.DataFrame(sf)

#df = sf


        
# ******************** NEW ***********************  
    


# =====  SELECT PLAYERS  =====
def select_view():

    df = sf
    # Print results.
    #for row in df.itertuples():
    #   st.write(f"{row.name} has a :{row.pet}:")
    #if st.checkbox('Show players data'):
    #    st.subheader('List of players')
    #    st.write(df)

    st.subheader('Who is playing today?')

    ## Display data table
    with st.expander("Modify Players/Scores"):

        original_df = df.copy()

        # Create an empty container for the data editor widget and message
        data_editor_container = st.empty()
        message_container = st.empty()

        # Display the initial data editor widget in the container
        with st.spinner('Loading...'):
            time.sleep(0.5)
            edited_df = data_editor_container.experimental_data_editor(original_df,use_container_width=True,num_rows="dynamic", key='data_editor')
            

        # Buttons to re-generate teams & go back to select players
        colApp, colDis, colBlnk2 = st.columns([4,4,10])

        with colApp:
            if st.button("Apply Changes"):
                df = edited_df
                message_container.success('Changes Applied!', icon="✅")
            
        with colDis:
            if st.button("Discard Changes"):
                # Refresh the original DataFrame and the data editor widget with a new key
                original_df = df.copy()

                edited_df = data_editor_container.experimental_data_editor(original_df, key='reset_editor')
                df = original_df
                
                # Replace the original data editor widget with the reset widget
                data_editor_container.empty()
                data_editor_container.experimental_data_editor(edited_df,use_container_width=True,num_rows="dynamic", key='data_editor_reset')
                message_container.info('Changes Discarded', icon="ℹ️")
        with colDis:
            st.write("")
            
        

    # Get the values from the first column of the DataFrame
    options = df.iloc[:, 0].unique()
    
    # Sort the values stored in options
    options = np.sort(options)

    # Create a multiselect widget and get the selected values, default is all players
    
    selected_values = st.multiselect("Players", options, options, label_visibility="collapsed")
    # Show number of players selected
    st.write("Number of players selected:", len(selected_values))

    # Create a new DataFrame containing only rows where the value in the first column is in selected_values
    pdfiltered = df[df.iloc[:, 0].isin(selected_values)]
        
    col1g, col2g, col3g = st.columns(3)

    with col1g:
        st.write(' ')

    with col2g:


        
        if st.button("GENERATE TEAMS",type="primary"):
            #players_selected = pdfiltered.copy()
            st.session_state.pdfiltered = pdfiltered
            st.session_state.view = "teams_view"
            st.experimental_rerun()

    with col3g:
        st.write(' ')
    



# =====  SHOW TEAMS  =====
def teams_view(players_selected):
    #st.write("teams_view")
    

    # --------------- GENERATE TEAMS -------------------------
    # Generate two equal teams based on the total points. 
    # Both Teams will have the same number of players or the difference will be 1 player.
    # Both teams will have the same total points or the difference will be the minimum possible
    # Both teams will have the same points by skill or the difference will be the minimum possible. 
    # All players must be assigned to a team.
    # There are 5 skills rated frorm 1 to 5 (1 is the lowest and 5 is the highest). Each skill rates are saved in columns 2 to 6.
    # The data structure is the first column is Name, the folowing are the skills: SPEED,SKILL ,Passing,Physical,Shooting
    # A summary of total points by team and by SKILL is shown at the end of the code.
    # This is a streamlit app.


    with st.spinner('Loading...'):
        #calculate total points by player
        players_selected['TotalPoints'] = players_selected.iloc[:, 1:6].sum(axis=1)

        # Define the fitness function
        def evaluate(individual):
            # Split the players into two teams based on the individual's binary representation
            team1 = players_selected.iloc[[i for i in range(len(individual)) if individual[i] == 0]]
            team2 = players_selected.iloc[[i for i in range(len(individual)) if individual[i] == 1]]

            # Calculate the difference in total points between the two teams
            diff_total_points = abs(sum(team1['TotalPoints']) - sum(team2['TotalPoints']))

            # Calculate the difference in total points by SKILL between the two teams
            diff_skill_points = sum([abs(sum(team1[SKILL]) - sum(team2[SKILL])) for SKILL in ['SPEED', 'SKILL', 'PASSING', 'PHYSICAL', 'SHOOTING']])

            # Return a tuple containing both differences (the fitness function will try to minimize these values)
            return (diff_total_points, diff_skill_points)

        # Set up the genetic algorithm
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,-1.0))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register('attr_bool', random.randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual,
                        toolbox.attr_bool, n=len(players_selected))
        toolbox.register('population', tools.initRepeat, list,
                        toolbox.individual)
        toolbox.register('evaluate', evaluate)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit,
                        indpb=0.05) 
        toolbox.register('select',
                        tools.selTournament,
                        tournsize=3)

        # Set up parameters for the genetic algorithm
        pop_size = 50 
        num_generations = 100 
        cx_prob = 0.5 
        mut_prob = 0.2 

        # Create initial population
        pop = toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate,pop))
        for ind ,fit in zip(pop ,fitnesses):
            ind.fitness.values=fit

        for g in range(num_generations):
            # Select parents for reproduction
            offspring=toolbox.select(pop,len(pop))
            
            # Clone selected individuals 
            offspring=list(map(toolbox.clone ,offspring))
            
            # Apply crossover and mutation to offspring
            
            # Crossover
            for child1 ,child2 in zip(offspring[::2],offspring[1::2]):
                if random.random()<cx_prob:
                    toolbox.mate(child1 ,child2)
                    del child1.fitness.values 
                    del child2.fitness.values
                    
                    
            # Mutation
            for mutant in offspring:
                if random.random()<mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
                    
            # Evaluate new individuals
            invalid_ind=[ind for ind in offspring if not ind.fitness.valid]
            fitnesses=map(toolbox.evaluate ,invalid_ind)
            for ind ,fit in zip(invalid_ind ,fitnesses):
                ind.fitness.values=fit
            
            
            # Replace population with new offspring
            pop[:]=offspring

        best_individuals=tools.selBest(pop,k=3)

        for bi in best_individuals:
            
            team1=pd.DataFrame(columns=['NAME','SPEED','SKILL','PASSING','PHYSICAL','SHOOTING'])
            team2=pd.DataFrame(columns=['NAME','SPEED','SKILL','PASSING','PHYSICAL','SHOOTING'])
            
            
            for i,val in enumerate(bi):
                player_row=pd.Series(players_selected.iloc[i])
                if val==0:
                    team1=team1.append(player_row ,ignore_index=True)
                else:
                    team2=team2.append(player_row ,ignore_index=True)


        # Create two columns to display the teams side-by-side
        team1_total_points=int(sum(team1['TotalPoints']))
        team2_total_points=int(sum(team2['TotalPoints']))


        # --------------- VIEW TEAMS -------------------------

        # Buttons to re-generate teams & go back to select players
        colb1, colb2, colb3 = st.columns([14,2,3])

        with colb1:
            st.write("")
        with colb2:
            if st.button("BACK"):
                st.session_state.view = "select_view"
                st.experimental_rerun()
        with colb3:    
            if st.button("GENERATE"):
                st.session_state.view = "teams_view"
                st.experimental_rerun()


        col1, col2 = st.columns(2)
        metrics_labels = ['SPEED', 'SKILL', 'PASSING', 'PHYSICAL', 'SHOOTING']

        with col1:
            st.markdown(f"<h3 style='color: white; background-color: #2c5fac; padding: 10px;'>Blue Team | {team1_total_points}</h3>", unsafe_allow_html=True)
            #st.metric(label="Total", value=team1_total_points, delta=team1_total_points-team2_total_points)
            
            metrics_values = [int(sum(team1[label])) for label in metrics_labels]
            metrics_deltas = [int(sum(team1[label])) - int(sum(team2[label])) for label in metrics_labels]

            colbl1, colbl2, colbl3, colbl4, colbl5 = st.columns(5)
            for i in range(5):
                with locals()[f"colbl{i+1}"]:
                    st.metric(label=metrics_labels[i].title(), value=metrics_values[i], delta=metrics_deltas[i])

            st.write(len(team1)," players")
            for name in team1.iloc[:, 0]:
                st.markdown(f"<img src='https://img.icons8.com/metro/20/2c5fac/user-male-circle.png'/> {name}", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<h3 style='color: white; background-color: #b80c09; padding: 10px;'>Red Team | {team2_total_points}</h3>", unsafe_allow_html=True)
            #st.metric(label="Total", value=team2_total_points, delta=team2_total_points-team1_total_points)
           
            metrics_values = [int(sum(team2[label])) for label in metrics_labels]
            metrics_deltas = [int(sum(team2[label])) - int(sum(team1[label])) for label in metrics_labels]

            colr1, colr2, colr3, colr4, colr5 = st.columns(5)
            for i in range(5):
                with locals()[f"colr{i+1}"]:
                    st.metric(label=metrics_labels[i].title(), value=metrics_values[i], delta=metrics_deltas[i])

            st.write(len(team2)," players")
            for name in team2.iloc[:, 0]:
                st.markdown(f"<img src='https://img.icons8.com/metro/20/b80c09/user-male-circle.png'/> {name}", unsafe_allow_html=True)

    
        team1["team"] = "blue"
        team2["team"] = "red"
        teams=pd.concat([team1 ,team2])

        #print(players_selected)

        # Remove the TotalPoints column
        #teams = teams.drop(columns=["TotalPoints"])

        # Group the data by team and attribute
        #teams_grouped = teams.groupby(['team']).sum()
        
        # Remove second column from teams_grouped
        #teams_grouped = teams_grouped.drop(columns=["Name"])

        

 
 
 


# =====  MAIN APP  =====
def app():

    if "view" not in st.session_state:
        st.session_state.view = "select_view"

    if "players_selected" not in st.session_state:
        st.session_state.players_selected = []

    if st.session_state.view == "select_view":
        select_view()
    elif st.session_state.view == "teams_view":
        players_selected = st.session_state.pdfiltered
        teams_view(players_selected)

if __name__ == "__main__":
    app() 