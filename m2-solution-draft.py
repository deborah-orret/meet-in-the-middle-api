import networkx as nx
import matplotlib.pyplot as plt
import csv
from math import radians, sin, cos, sqrt, atan2
import random
import pandas as pd

# Create a graph object
G = nx.Graph()
# Add nodes (stops)
stops = [
    # A - Line - 2 min per stop 
    "1AVE/1ST", "2AVE/1ST", "3AVE/1ST", "4AVE/1ST", "5AVE/1ST", "5AVE/5ST", "5AVE/10ST", "5AVE/15ST", "5AVE/20ST", "5AVE/25ST", "5AVE/30ST", "5AVE/35ST",
    "5AVE/40ST", "5AVE/45ST", "5AVE/50ST", "5AVE/55ST", "5AVE/60ST", "5AVE/65ST", "5AVE/70ST", "5AVE/75ST", "5AVE/80ST", 
    "6AVE/80ST", "7AVE/80ST", "8AVE/80ST", "9AVE/80ST", "10AVE/80ST", "11AVE/80ST", "12AVE/80ST", "13AVE/80ST", "14AVE/80ST", "15AVE/80ST", "16AVE/80ST", "17AVE/80ST",
    "18AVE/80ST", "19AVE/80ST", "20AVE/80ST",

    #M-Line - 2 min per stop
    "10AVE/10ST", "10AVE/20ST", "10AVE/30ST", "10AVE/40ST", "10AVE/50ST", "10AVE/60ST", "10AVE/70ST",

    # B - Line - 2 min per stop
    "1AVE/10ST", "2AVE/10ST", "3AVE/10ST", "4AVE/10ST",
    "6AVE/10ST", "7AVE/10ST", "8AVE/10ST", "9AVE/10ST",
    "11AVE/10ST", "12AVE/10ST", "13AVE/10ST", "14AVE/10ST", 
    "16AVE/10ST", "17AVE/10ST", "18AVE/10ST", "19AVE/10ST", "20AVE/10ST"

     # C - Line - 2 min per stop
    "1AVE/20ST", "2AVE/20ST", "3AVE/20ST", "4AVE/20ST",
    "6AVE/20ST", "7AVE/20ST", "8AVE/20ST", "9AVE/20ST",
    "11AVE/20ST", "12AVE/20ST", "13AVE/20ST", "14AVE/20ST", 
    "16AVE/20ST", "17AVE/20ST", "18AVE/20ST", "19AVE/20ST", "20AVE/20ST"

     # D - Line - 2 min per stop
    "1AVE/30ST", "2AVE/30ST", "3AVE/30ST", "4AVE/30ST",
    "6AVE/30ST", "7AVE/30ST", "8AVE/30ST", "9AVE/30ST",
    "11AVE/30ST", "12AVE/30ST", "13AVE/30ST", "14AVE/30ST", 
    "16AVE/30ST", "17AVE/30ST", "18AVE/30ST", "19AVE/30ST", "20AVE/30ST"

    # E - Line - 2 min per stop
    "1AVE/40ST", "2AVE/40ST", "3AVE/40ST", "4AVE/40ST",
    "6AVE/40ST", "7AVE/40ST", "8AVE/40ST", "9AVE/40ST",
    "11AVE/40ST", "12AVE/40ST", "13AVE/40ST", "14AVE/40ST", 
    "16AVE/40ST", "17AVE/40ST", "18AVE/40ST", "19AVE/40ST", "20AVE/40ST"

    # F - Line - 2 min per stop
    "1AVE/50ST", "2AVE/50ST", "3AVE/50ST", "4AVE/50ST",
    "6AVE/50ST", "7AVE/50ST", "8AVE/50ST", "9AVE/50ST",
    "11AVE/50ST", "12AVE/50ST", "13AVE/50ST", "14AVE/50ST", 
    "16AVE/50ST", "17AVE/50ST", "18AVE/50ST", "19AVE/50ST", "20AVE/50ST"

    # G - Line - 2 min per stop
    "1AVE/60ST", "2AVE/60ST", "3AVE/60ST", "4AVE/60ST",
    "6AVE/60ST", "7AVE/60ST", "8AVE/60ST", "9AVE/60ST",
    "11AVE/60ST", "12AVE/60ST", "13AVE/60ST", "14AVE/60ST", 
    "16AVE/60ST", "17AVE/60ST", "18AVE/60ST", "19AVE/60ST", "20AVE/60ST"

    # H - Line - 2 min per stop
    "1AVE/70ST", "2AVE/70ST", "3AVE/70ST", "4AVE/70ST",
    "6AVE/70ST", "7AVE/70ST", "8AVE/70ST", "9AVE/70ST",
    "11AVE/70ST", "12AVE/70ST", "13AVE/70ST", "14AVE/70ST", 
    "16AVE/70ST", "17AVE/70ST", "18AVE/70ST", "19AVE/70ST", "20AVE/70ST"

    #I-Line
    "20AVE/1ST", "19AVE/1ST", "18AVE/1ST", "17AVE/1ST", "16AVE/1ST", "15AVE/1ST",
    "15AVE/5ST", "15AVE/10ST", "15AVE/15ST", "15AVE/20ST", "15AVE/25ST", "15AVE/30ST", 
    "15AVE/35ST", "15AVE/40ST", "15AVE/45ST", "15AVE/50ST", "15AVE/55ST", "15AVE/60ST", 
    "15AVE/65ST", "15AVE/70ST", "15AVE/75ST"
]

G.add_nodes_from(stops)
# Add edges (connections with distances)
edges = [

# A-Line
    ("1AVE/1ST", "2AVE/1ST", 2), ("2AVE/1ST", "3AVE/1ST", 2), ("3AVE/1ST", "4AVE/1ST", 2),
    ("4AVE/1ST", "5AVE/1ST", 2), ("5AVE/1ST", "5AVE/5ST", 2), ("5AVE/5ST", "5AVE/10ST", 2),
    ("5AVE/10ST", "5AVE/15ST", 2), ("5AVE/15ST", "5AVE/20ST", 2), ("5AVE/20ST", "5AVE/25ST", 2),
    ("5AVE/25ST", "5AVE/30ST", 2), ("5AVE/30ST", "5AVE/35ST", 2), ("5AVE/35ST", "5AVE/40ST", 2),
    ("5AVE/40ST", "5AVE/45ST", 2), ("5AVE/45ST", "5AVE/50ST", 2), ("5AVE/50ST", "5AVE/55ST", 2),
    ("5AVE/55ST", "5AVE/60ST", 2), ("5AVE/60ST", "5AVE/65ST", 2), ("5AVE/65ST", "5AVE/70ST", 2),
    ("5AVE/70ST", "5AVE/75ST", 2), ("5AVE/75ST", "5AVE/80ST", 2),
    ("5AVE/80ST", "6AVE/80ST", 2), ("6AVE/80ST", "7AVE/80ST", 2), ("7AVE/80ST", "8AVE/80ST", 2),
    ("8AVE/80ST", "9AVE/80ST", 2), ("9AVE/80ST", "10AVE/80ST", 2), ("10AVE/80ST", "11AVE/80ST", 2),
    ("11AVE/80ST", "12AVE/80ST", 2), ("12AVE/80ST", "13AVE/80ST", 2), ("13AVE/80ST", "14AVE/80ST", 2),
    ("14AVE/80ST", "15AVE/80ST", 2), ("15AVE/80ST", "16AVE/80ST", 2), ("16AVE/80ST", "17AVE/80ST", 2),
    ("17AVE/80ST", "18AVE/80ST", 2), ("18AVE/80ST", "19AVE/80ST", 2), ("19AVE/80ST", "20AVE/80ST", 2),

    # M-Line
    ("10AVE/10ST", "10AVE/20ST", 2), ("10AVE/20ST", "10AVE/30ST", 2),
    ("10AVE/30ST", "10AVE/40ST", 2), ("10AVE/40ST", "10AVE/50ST", 2),
    ("10AVE/50ST", "10AVE/60ST", 2), ("10AVE/60ST", "10AVE/70ST", 2),

    # B-Line
    ("1AVE/10ST", "2AVE/10ST", 2), ("2AVE/10ST", "3AVE/10ST", 2), ("3AVE/10ST", "4AVE/10ST", 2), ("4AVE/10ST", "5AVE/10ST", 2), ("5AVE/10ST", "6AVE/10ST", 2),
    ("6AVE/10ST", "7AVE/10ST", 2), ("7AVE/10ST", "8AVE/10ST", 2),
    ("8AVE/10ST", "9AVE/10ST", 2), ("9AVE/10ST", "10AVE/10ST", 2), ("10AVE/10ST", "11AVE/10ST", 2), ("11AVE/10ST", "12AVE/10ST", 2),
    ("12AVE/10ST", "13AVE/10ST", 2), ("13AVE/10ST", "14AVE/10ST", 2), ("14AVE/10ST", "15AVE/10ST", 2), ("15AVE/10ST", "16AVE/10ST", 2),
    ("16AVE/10ST", "17AVE/10ST", 2), ("17AVE/10ST", "18AVE/10ST", 2), ("18AVE/10ST", "19AVE/10ST", 2),
    ("19AVE/10ST", "20AVE/10ST", 2),

    # C-Line
    ("1AVE/20ST", "2AVE/20ST", 2), ("2AVE/20ST", "3AVE/20ST", 2), ("3AVE/20ST", "4AVE/20ST", 2),
    ("4AVE/20ST", "5AVE/20ST", 2), ("5AVE/20ST", "6AVE/20ST", 2), ("6AVE/20ST", "7AVE/20ST", 2), ("7AVE/20ST", "8AVE/20ST", 2),
    ("8AVE/20ST", "9AVE/20ST", 2), ("9AVE/20ST", "10AVE/20ST", 2), ("10AVE/20ST", "11AVE/20ST", 2), ("11AVE/20ST", "12AVE/20ST", 2),
    ("12AVE/20ST", "13AVE/20ST", 2), ("13AVE/20ST", "14AVE/20ST", 2), ("14AVE/20ST", "15AVE/20ST", 2), ("15AVE/20ST", "16AVE/20ST", 2),
    ("16AVE/20ST", "17AVE/20ST", 2), ("17AVE/20ST", "18AVE/20ST", 2), ("18AVE/20ST", "19AVE/20ST", 2),
    ("19AVE/20ST", "20AVE/20ST", 2),

    # D-Line
    ("1AVE/30ST", "2AVE/30ST", 2), ("2AVE/30ST", "3AVE/30ST", 2), ("3AVE/30ST", "4AVE/30ST", 2),
    ("4AVE/30ST", "5AVE/30ST", 2), ("5AVE/30ST", "6AVE/30ST", 2), ("6AVE/30ST", "7AVE/30ST", 2), ("7AVE/30ST", "8AVE/30ST", 2),
    ("8AVE/30ST", "9AVE/30ST", 2), ("9AVE/30ST", "10AVE/30ST", 2), ("10AVE/30ST", "11AVE/30ST", 2), ("11AVE/30ST", "12AVE/30ST", 2),
    ("12AVE/30ST", "13AVE/30ST", 2), ("13AVE/30ST", "14AVE/30ST", 2), ("14AVE/30ST", "15AVE/30ST", 2), ("15AVE/30ST", "16AVE/30ST", 2),
    ("16AVE/30ST", "17AVE/30ST", 2), ("17AVE/30ST", "18AVE/30ST", 2), ("18AVE/30ST", "19AVE/30ST", 2),
    ("19AVE/30ST", "20AVE/30ST", 2),

    # E-Line
    ("1AVE/40ST", "2AVE/40ST", 2), ("2AVE/40ST", "3AVE/40ST", 2), ("3AVE/40ST", "4AVE/40ST", 2),
    ("4AVE/40ST", "5AVE/40ST", 2), ("5AVE/40ST", "6AVE/40ST", 2), ("6AVE/40ST", "7AVE/40ST", 2), ("7AVE/40ST", "8AVE/40ST", 2),
    ("8AVE/40ST", "9AVE/40ST", 2), ("9AVE/40ST", "10AVE/40ST", 2), ("10AVE/40ST", "11AVE/40ST", 2), ("11AVE/40ST", "12AVE/40ST", 2),
    ("12AVE/40ST", "13AVE/40ST", 2), ("13AVE/40ST", "14AVE/40ST", 2), ("14AVE/40ST", "15AVE/40ST", 2), ("15AVE/40ST", "16AVE/40ST", 2),
    ("16AVE/40ST", "17AVE/40ST", 2), ("17AVE/40ST", "18AVE/40ST", 2), ("18AVE/40ST", "19AVE/40ST", 2),
    ("19AVE/40ST", "20AVE/40ST", 2),

    # F-Line
    ("1AVE/50ST", "2AVE/50ST", 2), ("2AVE/50ST", "3AVE/50ST", 2), ("3AVE/50ST", "4AVE/50ST", 2),
    ("4AVE/50ST", "5AVE/50ST", 2), ("5AVE/50ST", "6AVE/50ST", 2), ("6AVE/50ST", "7AVE/50ST", 2), ("7AVE/50ST", "8AVE/50ST", 2),
    ("8AVE/50ST", "9AVE/50ST", 2), ("9AVE/50ST", "10AVE/50ST", 2), ("10AVE/50ST", "11AVE/50ST", 2),("11AVE/50ST", "12AVE/50ST", 2),
    ("12AVE/50ST", "13AVE/50ST", 2), ("13AVE/50ST", "14AVE/50ST", 2),
    ("14AVE/50ST", "15AVE/50ST", 2), ("15AVE/50ST", "16AVE/50ST", 2), ("16AVE/50ST", "17AVE/50ST", 2),
    ("17AVE/50ST", "18AVE/50ST", 2), ("18AVE/50ST", "19AVE/50ST", 2),
    ("19AVE/50ST", "20AVE/50ST", 2),

    # G-Line
    ("1AVE/60ST", "2AVE/60ST", 2), ("2AVE/60ST", "3AVE/60ST", 2), ("3AVE/60ST", "4AVE/60ST", 2),
    ("4AVE/60ST", "5AVE/60ST", 2), ("5AVE/10ST", "6AVE/10ST", 2), ("6AVE/60ST", "7AVE/60ST", 2), ("7AVE/60ST", "8AVE/60ST", 2),
    ("8AVE/60ST", "9AVE/60ST", 2), ("9AVE/60ST", "10AVE/60ST", 2), ("10AVE/60ST", "11AVE/60ST", 2), ("11AVE/60ST", "12AVE/60ST", 2),
    ("12AVE/60ST", "13AVE/60ST", 2), ("13AVE/60ST", "14AVE/60ST", 2),
    ("14AVE/60ST", "15AVE/60ST", 2), ("15AVE/60ST", "16AVE/60ST", 2), ("16AVE/60ST", "17AVE/60ST", 2),
    ("17AVE/60ST", "18AVE/60ST", 2), ("18AVE/60ST", "19AVE/60ST", 2),
    ("19AVE/60ST", "20AVE/60ST", 2),

    # H-Line
    ("1AVE/70ST", "2AVE/70ST", 2), ("2AVE/70ST", "3AVE/70ST", 2), ("3AVE/70ST", "4AVE/70ST", 2),
    ("4AVE/70ST", "5AVE/70ST", 2), ("5AVE/70ST", "6AVE/70ST", 2), ("6AVE/70ST", "7AVE/70ST", 2), ("7AVE/70ST", "8AVE/70ST", 2),
    ("8AVE/70ST", "9AVE/70ST", 2), ("9AVE/70ST", "10AVE/70ST", 2), ("10AVE/70ST", "11AVE/70ST", 2), ("11AVE/70ST", "12AVE/70ST", 2),
    ("12AVE/70ST", "13AVE/70ST", 2), ("13AVE/70ST", "14AVE/70ST", 2),
    ("14AVE/70ST", "15AVE/70ST", 2), ("15AVE/70ST", "16AVE/70ST", 2), ("16AVE/70ST", "17AVE/70ST", 2),
    ("17AVE/70ST", "18AVE/70ST", 2), ("18AVE/70ST", "19AVE/70ST", 2),
    ("19AVE/70ST", "20AVE/70ST", 2),

    # I-Line
    ("20AVE/1ST", "19AVE/1ST", 2), ("19AVE/1ST", "18AVE/1ST", 2), ("18AVE/1ST", "17AVE/1ST", 2),
    ("17AVE/1ST", "16AVE/1ST", 2), ("16AVE/1ST", "15AVE/1ST", 2),
    ("15AVE/1ST", "15AVE/5ST", 2), ("15AVE/5ST", "15AVE/10ST", 2),
    ("15AVE/10ST", "15AVE/15ST", 2), ("15AVE/15ST", "15AVE/20ST", 2),
    ("15AVE/20ST", "15AVE/25ST", 2), ("15AVE/25ST", "15AVE/30ST", 2),
    ("15AVE/30ST", "15AVE/35ST", 2), ("15AVE/35ST", "15AVE/40ST", 2),
    ("15AVE/40ST", "15AVE/45ST", 2), ("15AVE/45ST", "15AVE/50ST", 2),
    ("15AVE/50ST", "15AVE/55ST", 2), ("15AVE/55ST", "15AVE/60ST", 2),
    ("15AVE/60ST", "15AVE/65ST", 2), ("15AVE/65ST", "15AVE/70ST", 2),
    ("15AVE/70ST", "15AVE/75ST", 2), ("15AVE/75ST", "15AVE/80ST", 2)
    
]

# Add edges to the graph with distances as weights
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

def find_shortest_path_and_time(G, start, end):
    try:
        # Calculate shortest path using Dijkstra's algorithm
        shortest_path = nx.shortest_path(G, source=start, target=end, weight='weight')
        
        # Calculate total travel time
        total_time = 0
        for i in range(len(shortest_path) - 1): # O(n)
            current_stop = shortest_path[i]
            next_stop = shortest_path[i + 1]
            total_time += G[current_stop][next_stop]['weight']
        
        return shortest_path, total_time
    
    except nx.NetworkXNoPath:
        return None, float('inf')  # Return None path and infinite time if no path exists

def haversine(lat1, lon1, lat2, lon2):
    """ Calculate the great circle distance between two points on the earth (specified in decimal degrees) """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Calculate the distance
    distance = R * c
    
    return distance

def nearest_node(coordinates, csv_file):
    # Initialize variables to keep track of the closest intersection and its distance
    closest_intersection = None
    min_distance = float('inf')
    
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        
        for row in reader:
            intersection = row[0]
            lat = float(row[1])
            lon = float(row[2])
            
            # Calculate distance between given coordinates and intersection coordinates
            distance = haversine(coordinates[0], coordinates[1], lat, lon)

            
            # Update closest intersection if this one is closer
            if distance < min_distance:
                min_distance = distance
                closest_intersection = intersection
    return closest_intersection, min_distance

def calculate_eta(location_1, location_2):
    
    nearest_intersection1, additional_distance1 = nearest_node(location_1, stops_csv) # O(n)
    nearest_intersection2, additional_distance2 = nearest_node(location_2, stops_csv) # O(n)

    # Calculate the amount of time it takes to take the metro
    shortest_path_metro, total_time_metro = find_shortest_path_and_time(G, nearest_intersection1, nearest_intersection2) # O(n)
    # Calculate the time it takes to walk to the nearest metro stops
    walking_time = (additional_distance2/5*60) + (additional_distance1/5*60)

    total_time_walking = (haversine(location_1[0], location_1[1], location_2[0], location_2[1])/5 * 60)


    if shortest_path_metro:
        if total_time_metro < total_time_walking:
            return total_time_metro + walking_time
        else:
            return total_time_walking
    else:
        return "No path found."
    
stops_csv = "/Users/deborahorret/workspaces/byteboard/repos/meet-in-the-middle-api/byte-bay-metro-stops.csv"

# Task 1

# Write a method that parses in the city activity csv file and filters it based on 1 or more given parameters 

restaurants = pd.read_csv("/Users/deborahorret/workspaces/byteboard/repos/meet-in-the-middle-api/restaurants.csv")

def filter_data(df, rating = None, price = None):
    filtered_df = df
    
    if rating is not None:
        filtered_df = filtered_df[filtered_df['Rating'] >= rating]
    
    if price is not None:
        filtered_df = filtered_df[filtered_df['Price Category'] <= price]
    
    return filtered_df

# edge case, what if the data set is empty?

# Task 2 ?

def get_etas(activities, user_1_coordinates, user_2_coordinates):
    eta_dataset = []
    for index, row in activities.iterrows():
        name = row['Name']
        activity_coordinates = ((row['Lat'], row['Lon']))
        eta1 = calculate_eta(user_1_coordinates, activity_coordinates) # this is the part you'd want to optimize to avoid having to call this API for every row
        eta2 = calculate_eta(user_2_coordinates, activity_coordinates) # currently at O(n)
        
        eta_dataset.append((name, eta1, eta2))
    
    return eta_dataset

def find_optimal_location(locations):
    # Sort locations based on eta_to_location1 + eta_to_location2
    sorted_locations = sorted(locations, key=lambda x: x[1] + x[2])

    min_total_time = float('inf')
    optimal_location = None

    for loc in sorted_locations:
        eta_to_location1 = loc[1]
        eta_to_location2 = loc[2]
        
        total_commute_time = eta_to_location1 + eta_to_location2
        fairness_metric = abs(eta_to_location1 - eta_to_location2)
        
        # Calculate combined score (total_commute_time + fairness_metric)
        score = total_commute_time + fairness_metric
        
        # Update optimal location if this location has a lower combined score
        if score < min_total_time:
            min_total_time = score
            optimal_location = loc
    
    return optimal_location


# Task 3 




print(find_optimal_location(get_etas(filter_data(restaurants, rating = 4.0, price = 1.0), (37.808769874598,-122.36074244076914), (37.85920488834141,-122.3903587567014))))
