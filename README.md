Optimizing Supply Chain Costs with Python and MILP
For this project, we worked on minimizing the total cost of producing and delivering vehicles from three manufacturing plants (Troy, Newark, and Harrisburg) to eight retail studios located across different cities in the U.S. We used Mixed-Integer Linear Programming (MILP) with Python and the PuLP library to build our optimization model.
 
Data
•	Monthly Demand Data for each studio (January–December)
•	Distances between plants and studios
•	Production Efficiency & Energy Costs at each plant and production line
•	Truck Capacity (250 units max per trip)
•	Time Limits (each plant can operate up to 240 hours a month)
 
How the Model Works
We set up decision variables to:
•	Track how many units are produced at each line
•	Allocate monthly production hours
•	Decide how many units to send from each plant to each studio
•	Determine whether a truck is needed for that route
The goal was to minimize the total cost, which includes:
•	Transportation cost (based on distance, truck use, and fuel)
•	Production cost (based on energy usage and plant electricity rates)
We ran this model month by month, updating the demand data to reflect changes over the year.
 
Results 
•	Optimized cost: $5,728.70
•	Initial cost without optimization: $12,690.00
•	Savings: About $6,961.30 or 55%
•	All studio demands were fully met while staying within production and shipping limits.
 
Graphs Created 
We used Plotly to build charts and visuals, including:
•	Bar chart showing how each plant contributed to studio demand
•	Heatmap of units shipped
•	Sankey diagram to show product flow
•	Pie chart for share of total units from each plant
•	Line chart comparing costs before and after optimization
This project shows how companies can save thousands just by making smarter production and shipping decisions. The same model can be scaled or adjusted for different months, cost changes, or even more plants and studios.


 
pip install pulp
In [3]:
pip install plotly

In [7]:
pip install notebook --upgrade

                                                                                                                                                 In [9]:
import pandas as pd
from pulp import LpMinimize, LpProblem, LpVariable, lpSum
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
In [11]:
#import data 
data=pd.ExcelFile("Project Data.xlsx")
In [13]:
#Retail Store Demand DataFrame 
demand_df = data.parse("Retail Store Demand")
demand_df.columns = demand_df.iloc[0]
demand_df=demand_df[1:].reset_index(drop=True)
demand_df.fillna(0, inplace=True)
demand_df 
Out[13]:
	Month	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford	NaN
0	January	200	100	125	250	225	400	50	75	0.000000
1	February	150	125	100	300	200	425	75	100	200.000000
2	March	225	150	75	250	225	375	100	125	155.000000
3	April	250	200	100	200	200	350	125	150	218.000000
4	May	250	175	75	200	250	300	75	125	246.800000
5	June	180	175	100	300	175	400	50	100	249.680000
6	July	180	200	125	250	200	250	100	125	186.968000
7	August	200	150	200	300	150	200	150	150	180.696800
8	September	150	100	150	350	200	225	25	100	198.069680
9	October	150	100	100	350	250	400	50	75	154.806968
10	November	200	75	150	400	300	500	150	100	150.480697
11	December	300	200	175	450	400	475	150	150	195.048070
In [17]:


df = pd.read_excel(data, sheet_name=1, header=None)
df.head(20)
Out[17]:
	0	1	2	3	4	5	6	7	8	9	10
0	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	Shipping Distances	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford	NaN	NaN
2	Troy, NY	503	477	293	235	174	157	168	117	NaN	NaN
3	Newark, NJ	360	499	286	88	225	12	190	127	NaN	NaN
4	Harrisburg, PA	203	327	293	106	385	169	349	286	NaN	NaN
5	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
6	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
7	Gallons for Round Trip (15 mpg)	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford	NaN	NaN
8	Troy, NY	67.066667	63.6	39.066667	31.333333	23.2	20.933333	22.4	15.6	NaN	NaN
9	Newark, NJ	48	66.533333	38.133333	11.733333	30	1.6	25.333333	16.933333	NaN	NaN
10	Harrisburg, PA	27.066667	43.6	39.066667	14.133333	51.333333	22.533333	46.533333	38.133333	NaN	NaN
11	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
12	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
13	Round Trip Costs per Truck: $4 per gallon	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford	NaN	NaN
14	Troy, NY	268.266667	254.4	156.266667	125.333333	92.8	83.733333	89.6	62.4	NaN	NaN
15	Newark, NJ	192	266.133333	152.533333	46.933333	120	6.4	101.333333	67.733333	NaN	NaN
16	Harrisburg, PA	108.266667	174.4	156.266667	56.533333	205.333333	90.133333	186.133333	152.533333	NaN	NaN
17	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
18	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
19	Shipping Costs Per	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford	NaN	Shipping Costs Per Unit ($)*
We have three data frames for Shipping Costs. Shipping Distance Data Frame Gallons for Round Trip (15 mpg) Data Frame Round Trip Costs per Truck Shipping Costs per Unit
In [19]:
#Shipping Distances Data Frame 
df.head()
df_distances = df.iloc[2:5, 1:9].copy()
df_distances.index = df.iloc[2:5, 0]
df_distances.columns = df.iloc[1, 1:9]
df_distances = df_distances.apply(pd.to_numeric, errors='coerce')
df_distances.isna().sum()
df_distances.dropna(how='any', inplace=True)
df_distances.columns.name="Shipping Distances"

print(df_distances.columns)
df_distances.index.name = ""

df_distances
Index(['Pittsburgh', 'Cleveland', 'Buffalo', 'Philadelphia', 'Boston',
       'New York', 'Providence', 'Hartford'],
      dtype='object', name='Shipping Distances')
Out[19]:
Shipping Distances	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford
								
Troy, NY	503	477	293	235	174	157	168	117
Newark, NJ	360	499	286	88	225	12	190	127
Harrisburg, PA	203	327	293	106	385	169	349	286
In [21]:
# Gallons for Round Trip Data Frame 
df_gallons = df.iloc[7:12,1:9].copy()
df_gallons.columns=df.iloc[7,1:9]
df_gallons.index = df.iloc[7:12, 0]
df_gallons = df_gallons.apply(pd.to_numeric, errors='coerce')
df_gallons.dropna(how='any', inplace=True)
df_gallons.columns.name="Gallons for Round Trip (15 mpg)"
df_gallons.index.name=""
df_gallons
Out[21]:
Gallons for Round Trip (15 mpg)	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford
								
Troy, NY	67.066667	63.600000	39.066667	31.333333	23.200000	20.933333	22.400000	15.600000
Newark, NJ	48.000000	66.533333	38.133333	11.733333	30.000000	1.600000	25.333333	16.933333
Harrisburg, PA	27.066667	43.600000	39.066667	14.133333	51.333333	22.533333	46.533333	38.133333
In [23]:
#3. Round Trip Costs per Truck 
df_costs = df.iloc[13:19, 1:9].copy()
df_costs.columns = df.iloc[13, 1:9]
df_costs.index = df.iloc[13:19, 0]
df_costs = df_costs.apply(pd.to_numeric, errors='coerce')
df_costs.dropna(how='any',inplace=True)
df_costs.columns.name="Round Trip Costs per Truck: $4 per gallon"
df_costs.index.name=""
df_costs
Round Trip Costs per Truck: $4 per gallon	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford
								
Troy, NY	268.266667	254.400000	156.266667	125.333333	92.800000	83.733333	89.600000	62.400000
Newark, NJ	192.000000	266.133333	152.533333	46.933333	120.000000	6.400000	101.333333	67.733333
Harrisburg, PA	108.266667	174.400000	156.266667	56.533333	205.333333	90.133333	186.133333	152.533333
Out[23]:
In [25]:
#Shipping costs per unit 
df_per_unit = df.iloc[19:26, 1:9].copy()
df_per_unit.columns = df.iloc[19, 1:9]
df_per_unit.index = df.iloc[19:26, 0]
df_per_unit = df_per_unit.apply(pd.to_numeric, errors='coerce')
df_per_unit.dropna(how='any',inplace=True)
df_per_unit.columns.name="Shipping Costs Per Unit"
df_per_unit.index.name=""
df_per_unit
Out[25]:
Shipping Costs Per Unit	Pittsburgh	Cleveland	Buffalo	Philadelphia	Boston	New York	Providence	Hartford
								
Troy, NY	1.073067	1.017600	0.625067	0.501333	0.371200	0.334933	0.358400	0.249600
Newark, NJ	0.768000	1.064533	0.610133	0.187733	0.480000	0.025600	0.405333	0.270933
Harrisburg, PA	0.433067	0.697600	0.625067	0.226133	0.821333	0.360533	0.744533	0.610133
Data Frames For Production Line Characteristics for Troy,NY, Newark,NJ, Harrisburg,PA
In [27]:
df_prod = pd.read_excel(data, sheet_name=2, header=None)
df_prod.head(5)
Out[27]:
	0	1	2	3	4	5	6
0	Production Line Characteristics	NaN	NaN	NaN	NaN	NaN	NaN
1	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	Troy, NY	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per kWh	Energy Cost per unit
3	Prod Line 1	6	200	0.008	33.333333	0.15	5
4	Prod Line 2	2	255	0.01	127.5	0	19.125
In [29]:
# Select the relevant production values from Troy (adjust the range if needed)
troy_prod = df_prod.iloc[3:6, 1:8].copy()  # 3:6 selects 3 rows (check if you need 3:7 or 3:6)

# Set the column names using the appropriate row (row 2) and columns 1 to 8
troy_prod.columns = df_prod.iloc[2, 1:8]

# Set the row index (Troy, NY and others)
troy_prod.index = df_prod.iloc[3:6, 0]

# Convert to numeric in case values are strings
troy_prod = troy_prod.apply(pd.to_numeric, errors='coerce')
troy_prod.fillna(0, inplace=True)
# Set column and index names
troy_prod.columns.name = "Troy, NY"
troy_prod.index.name = ""

# Display result
troy_prod
Out[29]:
Troy, NY	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per kWh	Energy Cost per unit
						
Prod Line 1	6	200	0.008	33.333333	0.15	5.000
Prod Line 2	2	255	0.010	127.500000	0.00	19.125
Prod Line 3	1	275	0.014	275.000000	0.00	41.250
In [31]:
# Select the relevant production values from Newark (adjust the range if needed)
newark_prod = df_prod.iloc[8:11, 1:8].copy()  

# Set the column names using the appropriate row (row 2) and columns 1 to 8
newark_prod.columns = df_prod.iloc[7, 1:8]


newark_prod.index = df_prod.iloc[8:11, 0]

# Convert to numeric in case values are strings
newark_prod = newark_prod.apply(pd.to_numeric, errors='coerce')
newark_prod.fillna(0, inplace=True)
# Set column and index names
newark_prod.columns.name = "Newark, NJ"
newark_prod.index.name = ""

# Display result
newark_prod
Out[31]:
Newark, NJ	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per kWh	Energy Cost per unit
						
Prod Line 1	5	210	0.005	42.0	0.18	7.56
Prod Line 2	2	255	0.010	127.5	0.00	22.95
Prod Line 3	1	275	0.014	275.0	0.00	49.50
In [33]:
# Select the relevant production values from Harrisburg (adjust the range if needed)
harrisburg_prod = df_prod.iloc[13:16, 1:8].copy()  

# Set the column names using the appropriate row (row 2) and columns 1 to 8
harrisburg_prod.columns = df_prod.iloc[12, 1:8]


harrisburg_prod.index = df_prod.iloc[13:16, 0]

# Convert to numeric in case values are strings
harrisburg_prod = harrisburg_prod.apply(pd.to_numeric, errors='coerce')
harrisburg_prod.fillna(0, inplace=True)
# Set column and index names
harrisburg_prod.columns.name = "Harrisburg, PA"
harrisburg_prod.index.name = ""

# Display result
harrisburg_prod
Out[33]:
Harrisburg, PA	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per kWh	Energy Cost per unit
						
Prod Line 1	4.0	220	0.009	55.000000	0.09	4.95
Prod Line 2	3.0	235	0.010	78.333333	0.00	7.05
Prod Line 3	0.5	300	0.015	600.000000	0.00	54.00
FastProd: Production Line Characteristics
In [35]:
fastprod_df = pd.read_excel(data, sheet_name=3, header=None)
fastprod_df
Out[35]:
	0	1	2	3	4	5	6	7
0	FastProd - Production Line Characteristics	NaN	NaN	NaN	NaN	NaN	NaN	NaN
1	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2	NaN	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per Unit, Plant 1	Energy Cost per Unit, Plant 2	Energy Cost per Unit, Plant 3
3	FastProd Characteristics	8	300	0.01	37.5	5.625	6.75	3.375
In [37]:
fast_prod = fastprod_df.iloc[3:4, 1:8].copy()  

# Set the column names using the appropriate row (row 2) and columns 1 to 8
fast_prod.columns = fastprod_df.iloc[2, 1:8]
fast_prod.index = fastprod_df.iloc[3:4, 0]

# Convert to numeric in case values are strings
fast_prod = fast_prod.apply(pd.to_numeric, errors='coerce')
fast_prod.fillna(0, inplace=True)
# Set column and index names
fast_prod.columns.name = "FastProd-Production Line Characteristics"
fast_prod.index.name = ""

# Display result
fast_prod
Out[37]:
FastProd-Production Line Characteristics	Units per hour	Energy (kWh) per hour	Emissions per product	Energy (kWh) per unit	Energy Cost per Unit, Plant 1	Energy Cost per Unit, Plant 2	Energy Cost per Unit, Plant 3
							
FastProd Characteristics	8	300	0.01	37.5	5.625	6.75	3.375
In [39]:
# Sum production across all months (columns) for each plant
hours_per_year = 160 * 12  # = 1920

plant_capacity = {
    "Troy, NY": troy_prod["Units per hour"].sum() * hours_per_year,
    "Newark, NJ": newark_prod["Units per hour"].sum() * hours_per_year,
    "Harrisburg, PA": harrisburg_prod["Units per hour"].sum() * hours_per_year
}
Lucid Optimization Code: Minimize the total cost of production and transportation of vehicles from 3 manufacturing plants to 8 sales studios, while satisfying demand, respecting production constraints, and utilizing available resources efficiently.
In [41]:
plants = list(plant_capacity.keys())
studios = list(df_per_unit.columns)
In [43]:
#Prepare cost matrix 
plant_name_map={"Troy,NY":"Troy",
               "Newark,NJ":"Newark",
               "Harrisburg,PA":"Harrisburg"}
In [45]:
print(df_per_unit.index.tolist())
['Troy, NY', 'Newark, NJ', 'Harrisburg, PA']
In [47]:
cost = {
    (plant, studio): df_per_unit.loc[plant, studio]
    for plant in plants for studio in studios
}
In [49]:
# Total Demand per Studio
studio_demand = demand_df[studios].sum().to_dict()
studio_demand
Out[49]:
{'Pittsburgh': 2435,
 'Cleveland': 1750,
 'Buffalo': 1475,
 'Philadelphia': 3600,
 'Boston': 2775,
 'New York': 4300,
 'Providence': 1100,
 'Hartford': 1375}
In [51]:
#Total current demand and capacity
total_demand = sum(studio_demand.values())
total_capacity = sum(plant_capacity.values())
In [53]:
scale_factor = total_capacity / total_demand
print(f"Scaling demand by factor: {scale_factor:.4f}")
Scaling demand by factor: 2.5008
In [55]:
studio_demand_scaled=studio_demand
In [57]:
#Define the LP Model 
model = LpProblem("MultiPlant_Distribution_Cost_Minimization",LpMinimize)
#Decision Variables
x=LpVariable.dicts("x",(plants,studios),lowBound=0,cat="Integer")
In [59]:
#Objective Function 
model += lpSum(cost[i,j]*x[i][j] for i in plants for j in studios)
In [61]:
#Demand Constraints 
for j in studios:
    model += lpSum(x[i][j] for i in plants) == studio_demand_scaled[j], f"Demand_at_{j}"
In [63]:
#Capacity Constraints 
for i in plants:
    model += lpSum(x[i][j] for j in studios) <= plant_capacity[i], f"Capacity_at_{i}"
In [65]:
#Solve
model.solve()
Welcome to the CBC MILP Solver 
Version: 2.10.3 
Build Date: Dec 15 2019 

command line - /opt/anaconda3/lib/python3.12/site-packages/pulp/apis/../solverdir/cbc/osx/i64/cbc /var/folders/0l/zp3wpc151dngqk6wn_r8cgz80000gn/T/3817609f238c42a88e53154745b409dc-pulp.mps -timeMode elapsed -branch -printingOptions all -solution /var/folders/0l/zp3wpc151dngqk6wn_r8cgz80000gn/T/3817609f238c42a88e53154745b409dc-pulp.sol (default strategy 1)
At line 2 NAME          MODEL
At line 3 ROWS
At line 16 COLUMNS
At line 137 RHS
At line 149 BOUNDS
At line 174 ENDATA
Problem MODEL has 11 rows, 24 columns and 48 elements
Coin0008I MODEL read with 0 errors
Option for timeMode changed from cpu to elapsed
Continuous objective value is 5728.7 - 0.00 seconds
Cgl0004I processed model has 11 rows, 24 columns (24 integer (0 of which binary)) and 48 elements
Cbc0038I Full problem 11 rows 24 columns, reduced to 0 rows 0 columns
Cbc0012I Integer solution of 5728.704 found by greedy equality after 0 iterations and 0 nodes (0.02 seconds)
Cbc0001I Search completed - best objective 5728.703999999912, took 0 iterations and 0 nodes (0.02 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Cuts at root node changed objective from 5728.7 to 5728.7
Probing was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Gomory was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Knapsack was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Clique was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
MixedIntegerRounding2 was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
FlowCover was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
TwoMirCuts was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
ZeroHalf was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)

Result - Optimal solution found

Objective value:                5728.70400000
Enumerated nodes:               0
Total iterations:               0
Time (CPU seconds):             0.00
Time (Wallclock seconds):       0.02

Option for printingOptions changed from normal to all
Total time (CPU seconds):       0.01   (Wallclock seconds):       0.03

Out[65]:
1
In [67]:
#Output Results
results = pd.DataFrame([
    {
        "Plant": i,
        "Studio": j,
        "Vehicles_Shipped": x[i][j].varValue,
        "Cost_per_Unit": cost[i, j],
        "Total_Cost": x[i][j].varValue * cost[i, j]
    }
    for i in plants for j in studios if x[i][j].varValue > 0
])
results["Vehicles_Shipped"] = results["Vehicles_Shipped"].round(0).astype(int)
results["Total_Cost"] = results["Cost_per_Unit"] * results["Vehicles_Shipped"]
results["Total_Cost"] = results["Total_Cost"].apply(lambda x: f"${x:,.2f}")

results
Out[67]:
	Plant	Studio	Vehicles_Shipped	Cost_per_Unit	Total_Cost
0	Troy, NY	Boston	2775	0.371200	$1,030.08
1	Troy, NY	Providence	1100	0.358400	$394.24
2	Troy, NY	Hartford	1375	0.249600	$343.20
3	Newark, NJ	Buffalo	1475	0.610133	$899.95
4	Newark, NJ	Philadelphia	3600	0.187733	$675.84
5	Newark, NJ	New York	4300	0.025600	$110.08
6	Harrisburg, PA	Pittsburgh	2435	0.433067	$1,054.52
7	Harrisburg, PA	Cleveland	1750	0.697600	$1,220.80
Results from the Optimization Model
In [69]:
actual_demand = results.groupby("Studio")["Vehicles_Shipped"].sum().to_dict()
studio_summary = pd.DataFrame.from_dict(actual_demand, orient='index', columns=["Total Units Received"]).reset_index().rename(columns={"index": "Studio"})
studio_summary
Out[69]:
	Studio	Total Units Received
0	Boston	2775
1	Buffalo	1475
2	Cleveland	1750
3	Hartford	1375
4	New York	4300
5	Philadelphia	3600
6	Pittsburgh	2435
7	Providence	1100
In [71]:
studio_plant_summary = results.groupby(["Studio", "Plant"])["Vehicles_Shipped"].sum().reset_index()
In [73]:
merged = pd.merge(studio_plant_summary, studio_summary, on="Studio")
merged["Percent Contribution"] = (merged["Vehicles_Shipped"] / merged["Total Units Received"]) * 100
In [75]:
# Pivot and merge percent contribution
plant_percent_pivot = merged.pivot(index="Studio", columns="Plant", values="Percent Contribution").fillna(0)
combined_summary = pd.merge(studio_summary, plant_percent_pivot, on="Studio")
combined_summary[combined_summary.columns[2:]] = combined_summary[combined_summary.columns[2:]].round(1)

for col in combined_summary.columns[2:]:
    combined_summary[col] = combined_summary[col].astype(str) + "%"

combined_summary
Out[75]:
	Studio	Total Units Received	Harrisburg, PA	Newark, NJ	Troy, NY
0	Boston	2775	0.0%	0.0%	100.0%
1	Buffalo	1475	0.0%	100.0%	0.0%
2	Cleveland	1750	100.0%	0.0%	0.0%
3	Hartford	1375	0.0%	0.0%	100.0%
4	New York	4300	0.0%	100.0%	0.0%
5	Philadelphia	3600	0.0%	100.0%	0.0%
6	Pittsburgh	2435	100.0%	0.0%	0.0%
7	Providence	1100	0.0%	0.0%	100.0%
In [77]:
pivot_table = results.pivot_table(
    index="Plant",
    columns="Studio",
    values="Vehicles_Shipped",
    aggfunc="sum",
    fill_value=0
)

pivot_table
Out[77]:
Studio	Boston	Buffalo	Cleveland	Hartford	New York	Philadelphia	Pittsburgh	Providence
Plant								
Harrisburg, PA	0	0	1750	0	0	0	2435	0
Newark, NJ	0	1475	0	0	4300	3600	0	0
Troy, NY	2775	0	0	1375	0	0	0	1100
In [79]:
used_units = results.groupby("Plant")["Vehicles_Shipped"].sum()
utilization_table = pd.DataFrame({
    "Capacity": pd.Series(plant_capacity),
    "Used Units": used_units
})
utilization_table["Utilization (%)"] = (utilization_table["Used Units"] / utilization_table["Capacity"]) * 100
utilization_table["Utilization (%)"] = utilization_table["Utilization (%)"].round(1)
utilization_table
Out[79]:
	Capacity	Used Units	Utilization (%)
Harrisburg, PA	14400.0	4185	29.1
Newark, NJ	15360.0	9375	61.0
Troy, NY	17280.0	5250	30.4
Charts for results
Stacked Bar Chart: % contribution per plant to each studio
In [81]:
plant_percent_raw = merged.pivot(index="Studio", columns="Plant", values="Percent Contribution").fillna(0)
combined_summary_plot = pd.merge(studio_summary, plant_percent_raw, on="Studio")
plant_columns = [col for col in combined_summary_plot.columns if col not in ["Studio", "Total Units Received"]]

# Plot % contribution by plant to each studio 
plt.figure(figsize=(10, 6))
combined_summary_plot.set_index("Studio")[plant_columns].plot(
    kind="bar", stacked=True, figsize=(10, 6), colormap="tab20"
)
plt.title("Percent Contribution by Plant to Each Studio")
plt.ylabel("Contribution (%)")
plt.xlabel("Studio")
plt.legend(title="Plant")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
<Figure size 1000x600 with 0 Axes>
 
Heatmap: Percent contribution per studio-plant pair
In [83]:
#Heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(
    combined_summary_plot.set_index("Studio")[plant_columns],  # numeric data here
    annot=True, fmt=".1f", cmap="YlGnBu"
)
plt.title("Plant % Contribution to Each Studio (Heatmap)")
plt.ylabel("Studio")
plt.xlabel("Plant")
plt.tight_layout()
plt.show()
 
In [85]:
# Prepare source, target, and value lists
plants_list = list(results["Plant"].unique())
studios_list = list(results["Studio"].unique())

# Define source-target-value mappings
sources = [plants_list.index(row['Plant']) for _, row in results.iterrows()]
targets = [len(plants_list) + studios_list.index(row['Studio']) for _, row in results.iterrows()]
values = results["Vehicles_Shipped"].tolist()
labels = plants_list + studios_list

# Add custom hover labels with comma formatting
hover_labels = [
    f"{row['Vehicles_Shipped']:,} units from {row['Plant']} to {row['Studio']}"
    for _, row in results.iterrows()
]

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        label=labels,
        pad=20,
        thickness=20,
        line=dict(color="black", width=0.5)
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        label=hover_labels,
        hovertemplate='%{label}<extra></extra>'
    )
)])

# Layout and display
fig.update_layout(
    title_text="Vehicle Flow from Plants to Studios (Annual Distribution)",
    font_size=10
)
fig.show()
 
In [87]:
(Plant Utilization Bar Chart)
utilization = {
    plant: results[results["Plant"] == plant]["Vehicles_Shipped"].sum() / plant_capacity[plant] * 100
    for plant in plant_capacity
}

# Create Series for plotting
util_series = pd.Series(utilization).round(1)

# Plot bar chart
ax = util_series.plot(kind="bar", color="purple", figsize=(8, 5))
plt.title("Capacity Utilization by Plant (%)")
plt.ylabel("Utilization (%)")
plt.xlabel("Plant")
plt.ylim(0, 100)  # Optional: set Y-axis to 100% max
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add data labels on top of bars
for i, (plant, value) in enumerate(util_series.items()):
    ax.text(i, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()



 
In [90]:
Pie Chart on Plant Utilization
results.groupby("Plant")["Vehicles_Shipped"].sum().plot(
    kind="pie", autopct="%1.1f%%", figsize=(6, 6), startangle=90
)
plt.title("Share of Total Shipments by Plant")
plt.ylabel("")
plt.tight_layout()
plt.show()
 
In [92]:
# Baseline: Maximum cost scenario (worst-case, without optimization)
baseline_costs = {}
for studio in studios:
    max_cost = max(df_per_unit[studio])
    baseline_costs[studio] = studio_demand[studio] * max_cost

baseline_total = sum(baseline_costs.values())

# Optimized: Calculate total optimized costs (from your results DataFrame)
optimized_total = sum(results["Vehicles_Shipped"] * results["Cost_per_Unit"])

# Cost savings calculation
cost_savings = baseline_total - optimized_total

# Summary DataFrame for visualization
summary_df = pd.DataFrame({
    'Scenario': ['Baseline', 'Optimized', 'Cost Savings'],
    'Cost': [baseline_total, optimized_total, cost_savings]
})
In [94]:
# Bar Chart for Cost Saving Analysis
plt.figure(figsize=(10, 6))
bars = plt.bar(summary_df['Scenario'], summary_df['Cost'], color=['red', 'green', 'blue'])
plt.title('Cost Comparison: Baseline vs Optimized Scenario')
plt.ylabel('Total Cost ($)')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding labels on bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f"${yval:,.2f}", ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.show()
 
Explanation: Baseline Cost: This represents a simplified "worst-case scenario," assuming the highest possible shipping cost to each destination. Optimized Cost: Directly from your model's results. Savings: Clearly visualized as the difference between these two scenarios.

Month To Month Optimization Model
# OBJECTIVE FUNCTION

# starting with january: reduce cost to produce and transport, minimization
# 3 factories (I), 3 production lines in each factory (L), and 8 retail stores to ship to (J)


# step 1: list decision variables
# 1. y[i,j], whether truck is going from factory i to retail store j, or 0 if not
# 2. t[i,l], number of hours at each production line l at each factory i
#--------------------------------------------------------------------------
# 3. a[i,l], number of units produced at each production line l at each factory i
# 4. m[i], total number of hours at each factory i
# 5. x[i,j], number of units sent from each factory i to each retail store j



# step 2: list constants
# 1. DIS[i,j], distance from each factory i to each retail store j
# 2. F, fuel cost per mile
# 3. EC[i], energy cost per kilowatt hour at each factory
# 4. E[i,l], energy per hour at each production line l at each factory i




# WHERE i ∈ [1, 2, 3] factories

#       l ∈ [1, 2, 3] production lines within each factory

#       j ∈ [1, 2, 3, 4, 5, 6, 7, 8] retail stores





# step 3: make the objective function 1:

#       i j            i j      
#     y[1,1] * 2 * DIS[1,1] * F     +      y[1,2] * 2 * DIS[1,2] * F     +
#     y[1,3] * 2 * DIS[1,3] * F     +      y[1,4] * 2 * DIS[1,4] * F     +
#     y[1,5] * 2 * DIS[1,5] * F     +      y[1,6] * 2 * DIS[1,6] * F     +
#     y[1,7] * 2 * DIS[1,7] * F     +      y[1,8] * 2 * DIS[1,8] * F     +

#     y[2,1] * 2 * DIS[2,1] * F     +      y[2,2] * 2 * DIS[2,2] * F     +
#     y[2,3] * 2 * DIS[2,3] * F     +      y[2,4] * 2 * DIS[2,4] * F     +
#     y[2,5] * 2 * DIS[2,5] * F     +      y[2,6] * 2 * DIS[2,6] * F     +
#     y[2,7] * 2 * DIS[2,7] * F     +      y[2,8] * 2 * DIS[2,8] * F     +

#     y[3,1] * 2 * DIS[3,1] * F     +      y[3,2] * 2 * DIS[3,2] * F     +
#     y[3,3] * 2 * DIS[3,3] * F     +      y[3,4] * 2 * DIS[3,4] * F     +
#     y[3,5] * 2 * DIS[3,5] * F     +      y[3,6] * 2 * DIS[3,6] * F     +
#     y[3,7] * 2 * DIS[3,7] * F     +      y[3,8] * 2 * DIS[3,8] * F     +

#        i       i l      i l
#     EC[1] * t[1,1] * E[1,1]     +      EC[1] * t[1,2] * E[1,2]     +
#     EC[1] * t[1,3] * E[1,3]     +     

#     EC[2] * t[2,1] * E[2,1]     +      EC[2] * t[2,2] * E[2,2]     +
#     EC[2] * t[2,3] * E[2,3]     +     

#     EC[3] * t[3,1] * E[3,1]     +      EC[3] * t[3,2] * E[3,2]     +
#     EC[3] * t[3,3] * E[3,3]           






# step 4: list out constraints:

#       i l                            j                                                   
# 1   a[1,1] + a[1,2] + a[1,3] >= (DEM[1] + DEM[2] + DEM[3] + DEM[4] + DEM[5] + DEM[6] + DEM[7] + DEM[8])/3
#     a[2,1] + a[2,2] + a[2,3] >= (DEM[1] + DEM[2] + DEM[3] + DEM[4] + DEM[5] + DEM[6] + DEM[7] + DEM[8])/3                                                                                      
#     a[3,1] + a[3,2] + a[3,3] >= (DEM[1] + DEM[2] + DEM[3] + DEM[4] + DEM[5] + DEM[6] + DEM[7] + DEM[8])/3

#       i
# 2.  m[1] <= H   (H = 240)
#     m[2] <= H
#     m[3] <= H
 
#       i l       i
# 3.  t[1,1] <= m[1]
#     t[1,2] <= m[1]
#     t[1,3] <= m[1]

#     t[2,1] <= m[2]
#     t[2,2] <= m[2]
#     t[2,3] <= m[2]

#     t[3,1] <= m[3]
#     t[3,2] <= m[3]
#     t[3,3] <= m[3]

#        i l      i l      i l
# 4.   a[1,1] = B[1,1] * t[1,1]
#      a[1,2] = B[1,2] * t[1,2]
#      a[1,3] = B[1,3] * t[1,3]
#
#      a[2,1] = B[2,1] * t[2,1]
#      a[2,2] = B[2,2] * t[2,2]
#      a[2,3] = B[2,3] * t[2,3]
#
#      a[3,1] = B[3,1] * t[3,1]
#      a[3,2] = B[3,2] * t[3,2]
#      a[3,3] = B[3,3] * t[3,3]

#        i j                                                                      i l
# 5.   x[1,1] + x[1,2] + x[1,3] + x[1,4] + x[1,5] + x[1,6] + x[1,7] + x[1,8] <= a[1,1] + a[1,2] + a[1,3]
#      x[2,1] + x[2,2] + x[2,3] + x[2,4] + x[2,5] + x[2,6] + x[2,7] + x[2,8] <= a[2,1] + a[2,2] + a[2,3]
#      x[3,1] + x[3,2] + x[3,3] + x[3,4] + x[3,5] + x[3,6] + x[3,7] + x[3,8] <= a[3,1] + a[3,2] + a[3,3]

#        i l       i l
# 6.   a[1,1] <= B[1,1] * H
#      a[1,2] <= B[1,2] * H
#      a[1,3] <= B[1,3] * H 
#
#      a[2,1] <= B[2,1] * H
#      a[2,2] <= B[2,2] * H
#      a[2,3] <= B[2,3] * H
#
#      a[3,1] <= B[3,1] * H
#      a[3,2] <= B[3,2] * H
#      a[3,3] <= B[3,3] * H

#        i j                           j
# 7.   x[1,1] + x[2,1] + x[3,1] >= DEM[1]
#      x[1,2] + x[2,2] + x[3,2] >= DEM[2]
#      x[1,3] + x[2,3] + x[3,3] >= DEM[3]
#      x[1,4] + x[2,4] + x[3,4] >= DEM[4]
#      x[1,5] + x[2,5] + x[3,5] >= DEM[5]
#      x[1,6] + x[2,6] + x[3,6] >= DEM[6]
#      x[1,7] + x[2,7] + x[3,7] >= DEM[7]
#      x[1,8] + x[2,8] + x[3,8] >= DEM[8]

#        i j            i j
# 8.   x[1,1] <= TC * y[1,1]
#      x[1,2] <= TC * y[1,2]
#      x[1,3] <= TC * y[1,3]
#      x[1,4] <= TC * y[1,4]
#      x[1,5] <= TC * y[1,5]
#      x[1,6] <= TC * y[1,6]
#      x[1,7] <= TC * y[1,7]
#      x[1,8] <= TC * y[1,8]
#
#      x[2,1] <= TC * y[2,1]
#      x[2,2] <= TC * y[2,2]
#      x[2,3] <= TC * y[2,3]
#      x[2,4] <= TC * y[2,4]
#      x[2,5] <= TC * y[2,5]
#      x[2,6] <= TC * y[2,6]
#      x[2,7] <= TC * y[2,7]
#      x[2,8] <= TC * y[2,8]
#
#      x[3,1] <= TC * y[3,1]
#      x[3,2] <= TC * y[3,2]
#      x[3,3] <= TC * y[3,3]
#      x[3,4] <= TC * y[3,4]
#      x[3,5] <= TC * y[3,5]
#      x[3,6] <= TC * y[3,6]
#      x[3,7] <= TC * y[3,7]
#      x[3,8] <= TC * y[3,8]

#
# 9.     i l                                                                                  j
#      a[1,1] + a[1,2] + a[1,3] + a[2,1] + a[2,2] + a[2,3] + a[3,1] + a[3,2] + a[3,3] >= (DEM[1] + DEM[2] + DEM[3] + DEM[4] + DEM[5] + DEM[6] + DEM[7] + DEM[8])



# step 5: solve optimization problem for each month, change DEM[1]-DEM[8] to match the month

# (below)

# step 6: consider that each month's demand at each store is dependent on time of the year, plus random noise. Account
#         for random noise with statistical methods to determine interval for decision variables.

# (below)
(START) RUN ONLY 1 BLOCK OF CODE BELOW FOR DEMAND MONTH¶
In [42]:
# January

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[0][0:8]  # demand for first row, January, for Pittsburg (j=1) thru Hartford (j=8)
DEMtotal = demand_df.iloc[0][0:8].sum()   

monthlabel = 'January'
In [44]:
# February

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[1][0:8]  # demand for second row, February, for Pittsburg (j=1) thru Hartford (j=8)
DEMtotal = demand_df.iloc[1][0:8].sum()   

monthlabel = 'February'
In [46]:
# March

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[2][0:8]  # demand for third row, March, for Pittsburg (j=1) thru Hartford (j=8)
DEMtotal = demand_df.iloc[2][0:8].sum()   

monthlabel = 'March'
In [48]:
# April

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[3][0:8] 
DEMtotal = demand_df.iloc[3][0:8].sum()   

monthlabel = 'April'
In [50]:
# May

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[4][0:8]  
DEMtotal = demand_df.iloc[4][0:8].sum()   

monthlabel = 'May'
In [52]:
# June

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[5][0:8]  
DEMtotal = demand_df.iloc[5][0:8].sum()   

monthlabel = 'June'
In [54]:
# July

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[6][0:8] 
DEMtotal = demand_df.iloc[6][0:8].sum()   

monthlabel = 'July'
In [56]:
# August

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[7][0:8]  
DEMtotal = demand_df.iloc[7][0:8].sum()   

monthlabel = 'August'
In [58]:
# September

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[8][0:8] 
DEMtotal = demand_df.iloc[8][0:8].sum()   

monthlabel = 'September'
In [60]:
# October

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[9][0:8]  
DEMtotal = demand_df.iloc[9][0:8].sum()   

monthlabel = 'October'
In [62]:
# November

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[10][0:8]  
DEMtotal = demand_df.iloc[10][0:8].sum()   

monthlabel = 'November'
In [64]:
# December

DEM1, DEM2, DEM3, DEM4, DEM5, DEM6, DEM7, DEM8 = demand_df.iloc[11][0:8]  
DEMtotal = demand_df.iloc[11][0:8].sum()   

monthlabel = 'December'
RUN ONLY 1 BLOCK OF CODE ABOVE FOR DEMAND MONTH¶
In [70]:
prob = LpProblem("Cost_Minimization", LpMinimize)
prob.solve()


# print results

print("Status:", LpStatus[prob.status])
print()
print("Solved Decision Values for ", monthlabel, ":", sep = "")
print()
print()

for v in prob.variables()[0:4]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[4:8]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[8:12]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[12:20]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[20:28]:
    print(v.name, '=', v.varValue)


print("\n")

for v in prob.variables()[28:36]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[36:39]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[39:42]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[42:45]:
    print(v.name, '=', v.varValue)

print("\n")


for v in prob.variables()[45:53]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[53:61]:
    print(v.name, '=', v.varValue)

print("\n")

for v in prob.variables()[61:69]:
    print(v.name, '=', v.varValue)

print(monthlabel)
#del prob
Welcome to the CBC MILP Solver 
Version: 2.10.3 
Build Date: Dec 15 2019 

command line - /opt/anaconda3/lib/python3.12/site-packages/pulp/apis/../solverdir/cbc/osx/i64/cbc /var/folders/0l/zp3wpc151dngqk6wn_r8cgz80000gn/T/2fe7166f4ce74b4ab455e9cd9fe179d4-pulp.mps -timeMode elapsed -branch -printingOptions all -solution /var/folders/0l/zp3wpc151dngqk6wn_r8cgz80000gn/T/2fe7166f4ce74b4ab455e9cd9fe179d4-pulp.sol (default strategy 1)
At line 2 NAME          MODEL
At line 3 ROWS
At line 5 COLUMNS
At line 7 RHS
At line 8 BOUNDS
At line 10 ENDATA
Problem MODEL has 0 rows, 1 columns and 0 elements
Coin0008I MODEL read with 0 errors
Option for timeMode changed from cpu to elapsed
Empty problem - 0 rows, 1 columns and 0 elements
Optimal - objective value 0
Optimal objective 0 - 0 iterations time 0.002
Option for printingOptions changed from normal to all
Total time (CPU seconds):       0.00   (Wallclock seconds):       0.00

Status: Optimal

Solved Decision Values for December:


__dummy = None






















December
In [ ]:
 
In [72]:
# extract constants from dataframes

# get fuel cost per mile F by calculating a few of the notes from the Excel, does not appear in dataframe

F = 4 * (1/15)           #  fuel cost per mile = ($4/gallon) * (1/15 gallons/mile) = $__/mile

# get DIS[i,j] from dataframe distance_df

DIS1_1, DIS1_2, DIS1_3, DIS1_4, DIS1_5, DIS1_6, DIS1_7, DIS1_8 = distance_df.iloc[0]    # from troy (i=1) to each retail (j=1 to 8)
DIS2_1, DIS2_2, DIS2_3, DIS2_4, DIS2_5, DIS2_6, DIS2_7, DIS2_8 = distance_df.iloc[1]    # from newark (i=2) to each retail (j=1 to 8)
DIS3_1, DIS3_2, DIS3_3, DIS3_4, DIS3_5, DIS3_6, DIS3_7, DIS3_8 = distance_df.iloc[2]    # from harrisburg (i=3) to each retail (j=1 to 8) 

# get B[i,l] from first column of dataframes troy_prod, newark_prod, and harrisburg_prod

B1_1, B1_2, B1_3 = troy_prod.iloc[:,0]          # troy, i = 1, prod lines 1-3, l=1 to 3
B2_1, B2_2, B2_3 = newark_prod.iloc[:,0]        # newark, i = 2
B3_1, B3_2, B3_3 = harrisburg_prod.iloc[:,0]    # harrisburg, i = 3

# get E[i,l] from second column of dataframes troy_prod, newark_prod, and harrisburg_prod

E1_1, E1_2, E1_3 = troy_prod.iloc[:,1]     # troy, i = 1, prod lines 1-3
E2_1, E2_2, E2_3 = newark_prod.iloc[:,1]   # newark, i = 2
E3_1, E3_2, E3_3 = harrisburg_prod.iloc[:,1]     # harrisburg, i = 3

# get EC[i] from fifth column of dataframes troy_prod, newark_prod, and harrisburg_prod

EC1 = troy_prod.iloc[0,4]     # troy, i = 1, for any production line (energy cost for line only appears at top, 0's are placeholders)
EC2 = newark_prod.iloc[0,4]   # newark, i = 2 (energy cost for line only appears at top, 0's are placeholders)
EC3 = harrisburg_prod.iloc[0,4]     # harrisburg, i = 3 (energy cost for line only appears at top, 0's are placeholders)

# Truck capacity TC is 250 per a note in Excel, doesn't appear in dataframe

TC = 250

# monthly limit for production hours is 240 per conversation with Lucid manager (8 hrs x 30 days)

H = 240



# ------------------------------------------------------------------------------------------------------------------------------------------



# set up problem

prob = LpProblem("Minimize_Transportation_and_Production_Costs", LpMinimize)

# set up a[i,l] decision variables

a1_1 = LpVariable("Units_Produced_1Troy_Line1", lowBound=0, cat='Integer')
a1_2 = LpVariable("Units_Produced_1Troy_Line2", lowBound=0, cat='Integer')
a1_3 = LpVariable("Units_Produced_1Troy_Line3", lowBound=0, cat='Integer')
a2_1 = LpVariable("Units_Produced_2Newark_Line1", lowBound=0, cat='Integer')
a2_2 = LpVariable("Units_Produced_2Newark_Line2", lowBound=0, cat='Integer')
a2_3 = LpVariable("Units_Produced_2Newark_Line3", lowBound=0, cat='Integer')
a3_1 = LpVariable("Units_Produced_3Harrisburg_Line1", lowBound=0, cat='Integer')
a3_2 = LpVariable("Units_Produced_3Harrisburg_Line2", lowBound=0, cat='Integer')
a3_3 = LpVariable("Units_Produced_3Harrisburg_Line3", lowBound=0, cat='Integer')

# set up t[i,l] decision variables

t1_1 = LpVariable("Production_Hours_1Troy_Line1", lowBound=0, cat='Integer')
t1_2 = LpVariable("Production_Hours_1Troy_Line2", lowBound=0, cat='Integer')
t1_3 = LpVariable("Production_Hours_1Troy_Line3", lowBound=0, cat='Integer')
t2_1 = LpVariable("Production_Hours_2Newark_Line1", lowBound=0, cat='Integer')
t2_2 = LpVariable("Production_Hours_2Newark_Line2", lowBound=0, cat='Integer')
t2_3 = LpVariable("Production_Hours_2Newark_Line3", lowBound=0, cat='Integer')
t3_1 = LpVariable("Production_Hours_3Harrisburg_Line1", lowBound=0, cat='Integer')
t3_2 = LpVariable("Production_Hours_3Harrisburg_Line2", lowBound=0, cat='Integer')
t3_3 = LpVariable("Production_Hours_3Harrisburg_Line3", lowBound=0, cat='Integer')

# set up m[i] decision variables

m1 = LpVariable("Production_Hours_1Troy", lowBound=0, cat='Integer')
m2 = LpVariable("Production_Hours_2Newark", lowBound = 0, cat='Integer')
m3 = LpVariable("Production_Hours_3Harrisburg", lowBound = 0, cat='Integer')

# set up x[i,j] decision variables

x1_1 = LpVariable("Units_Sent_1Troy_to_1Pittsburgh", lowBound=0, cat='Integer')
x1_2 = LpVariable("Units_Sent_1Troy_to_2Cleveland", lowBound=0, cat='Integer')
x1_3 = LpVariable("Units_Sent_1Troy_to_3Buffalo", lowBound=0, cat='Integer')
x1_4 = LpVariable("Units_Sent_1Troy_to_4Philadelphia", lowBound=0, cat='Integer')
x1_5 = LpVariable("Units_Sent_1Troy_to_5Boston", lowBound=0, cat='Integer')
x1_6 = LpVariable("Units_Sent_1Troy_to_6NewYork", lowBound=0, cat='Integer')
x1_7 = LpVariable("Units_Sent_1Troy_to_7Providence", lowBound=0, cat='Integer')
x1_8 = LpVariable("Units_Sent_1Troy_to_8Hartford", lowBound=0, cat='Integer')
x2_1 = LpVariable("Units_Sent_2Newark_to_1Pittsburgh", lowBound=0, cat='Integer')
x2_2 = LpVariable("Units_Sent_2Newark_to_2Cleveland", lowBound=0, cat='Integer')
x2_3 = LpVariable("Units_Sent_2Newark_to_3Buffalo", lowBound=0, cat='Integer')
x2_4 = LpVariable("Units_Sent_2Newark_to_4Philadelphia", lowBound=0, cat='Integer')
x2_5 = LpVariable("Units_Sent_2Newark_to_5Boston", lowBound=0, cat='Integer')
x2_6 = LpVariable("Units_Sent_2Newark_to_6NewYork", lowBound=0, cat='Integer')
x2_7 = LpVariable("Units_Sent_2Newark_to_7Providence", lowBound=0, cat='Integer')
x2_8 = LpVariable("Units_Sent_2Newark_to_8Hartford", lowBound=0, cat='Integer')
x3_1 = LpVariable("Units_Sent_3Harrisburg_to_1Pittsburgh", lowBound=0, cat='Integer')
x3_2 = LpVariable("Units_Sent_3Harrisburg_to_2Cleveland", lowBound=0, cat='Integer')
x3_3 = LpVariable("Units_Sent_3Harrisburg_to_3Buffalo", lowBound=0, cat='Integer')
x3_4 = LpVariable("Units_Sent_3Harrisburg_to_4Philadelphia", lowBound=0, cat='Integer')
x3_5 = LpVariable("Units_Sent_3Harrisburg_to_5Boston", lowBound=0, cat='Integer')
x3_6 = LpVariable("Units_Sent_3Harrisburg_to_6NewYork", lowBound=0, cat='Integer')
x3_7 = LpVariable("Units_Sent_3Harrisburg_to_7Providence", lowBound=0, cat='Integer')
x3_8 = LpVariable("Units_Sent_3Harrisburg_to_8Hartford", lowBound=0, cat='Integer')

# set up y[i,j] decision variables

y1_1 = LpVariable("Truck_Sent_from_1Troy_to_1Pittsburgh?", lowBound=0, upBound=1, cat='Integer')
y1_2 = LpVariable("Truck_Sent_from_1Troy_to_2Cleveland?", lowBound=0, upBound=1, cat='Integer')
y1_3 = LpVariable("Truck_Sent_from_1Troy_to_3Buffalo?", lowBound=0, upBound=1, cat='Integer')
y1_4 = LpVariable("Truck_Sent_from_1Troy_to_4Philadelphia?", lowBound=0, upBound=1, cat='Integer')
y1_5 = LpVariable("Truck_Sent_from_1Troy_to_5Boston?", lowBound=0, upBound=1, cat='Integer')
y1_6 = LpVariable("Truck_Sent_from_1Troy_to_6NewYork?", lowBound=0, upBound=1, cat='Integer')
y1_7 = LpVariable("Truck_Sent_from_1Troy_to_7Providence?", lowBound=0, upBound=1, cat='Integer')
y1_8 = LpVariable("Truck_Sent_from_1Troy_to_8Hartford?", lowBound=0, upBound=1, cat='Integer')
y2_1 = LpVariable("Truck_Sent_from_2Newark_to_1Pittsburgh?", lowBound=0, upBound=1, cat='Integer')
y2_2 = LpVariable("Truck_Sent_from_2Newark_to_2Cleveland?", lowBound=0, upBound=1, cat='Integer')
y2_3 = LpVariable("Truck_Sent_from_2Newark_to_3Buffalo?", lowBound=0, upBound=1, cat='Integer')
y2_4 = LpVariable("Truck_Sent_from_2Newark_to_4Philadelphia?", lowBound=0, upBound=1, cat='Integer')
y2_5 = LpVariable("Truck_Sent_from_2Newark_to_5Boston?", lowBound=0, upBound=1, cat='Integer')
y2_6 = LpVariable("Truck_Sent_from_2Newark_to_6NewYork?", lowBound=0, upBound=1, cat='Integer')
y2_7 = LpVariable("Truck_Sent_from_2Newark_to_7Providence?", lowBound=0, upBound=1, cat='Integer')
y2_8 = LpVariable("Truck_Sent_from_2Newark_to_8Hartford?", lowBound=0, upBound=1, cat='Integer')
y3_1 = LpVariable("Truck_Sent_from_3Harrisburg_to_1Pittsburgh?", lowBound=0, upBound=1, cat='Integer')
y3_2 = LpVariable("Truck_Sent_from_3Harrisburg_to_2Cleveland?", lowBound=0, upBound=1, cat='Integer')
y3_3 = LpVariable("Truck_Sent_from_3Harrisburg_to_3Buffalo?", lowBound=0, upBound=1, cat='Integer')
y3_4 = LpVariable("Truck_Sent_from_3Harrisburg_to_4Philadelphia?", lowBound=0, upBound=1, cat='Integer')
y3_5 = LpVariable("Truck_Sent_from_3Harrisburg_to_5Boston?", lowBound=0, upBound=1, cat='Integer')
y3_6 = LpVariable("Truck_Sent_from_3Harrisburg_to_6NewYork?", lowBound=0, upBound=1, cat='Integer')
y3_7 = LpVariable("Truck_Sent_from_3Harrisburg_to_7Providence?", lowBound=0, upBound=1, cat='Integer')
y3_8 = LpVariable("Truck_Sent_from_3Harrisburg_to_8Hartford?", lowBound=0, upBound=1, cat='Integer')



# ------------------------------------------------------------------------------------------------------------------------------------------



# set up objective function:

prob += (y1_1 * 2 * DIS1_1 * F) + (y1_2 * 2 * DIS1_2 * F) + \
(y1_3 * 2 * DIS1_3 * F) + (y1_4 * 2 * DIS1_4 * F) + \
(y1_5 * 2 * DIS1_5 * F) + (y1_6 * 2 * DIS1_6 * F) + \
(y1_7 * 2 * DIS1_7 * F) + (y1_8 * 2 * DIS1_8 * F) + \
(y2_1 * 2 * DIS2_1 * F) + (y2_2 * 2 * DIS2_2 * F) + \
(y2_3 * 2 * DIS2_3 * F) + (y2_4 * 2 * DIS2_4 * F) + \
(y2_5 * 2 * DIS2_5 * F) + (y2_6 * 2 * DIS2_6 * F) + \
(y2_7 * 2 * DIS2_7 * F) + (y2_8 * 2 * DIS2_8 * F) + \
(y3_1 * 2 * DIS3_1 * F) + (y3_2 * 2 * DIS3_2 * F) + \
(y3_3 * 2 * DIS3_3 * F) + (y3_4 * 2 * DIS3_4 * F) + \
(y3_5 * 2 * DIS3_5 * F) + (y3_6 * 2 * DIS3_6 * F) + \
(y3_7 * 2 * DIS3_7 * F) + (y3_8 * 2 * DIS3_8 * F) + \
(EC1 * t1_1 * E1_1) + (EC1 * t1_2 * E1_2) + \
(EC1 * t1_3 * E1_3) + \
(EC2 * t2_1 * E2_1) + (EC2 * t2_2 * E2_2) + \
(EC2 * t2_3 * E2_3) + \
(EC3 * t3_1 * E3_1) + (EC3 * t3_2 * E3_2) + \
(EC3 * t3_3 * E3_3)



# ------------------------------------------------------------------------------------------------------------------------------------------



# set up constraints:

# each individual factory needs to produce at least 1/3 of total monthly demand [sum of lines' production >= 1/3 x sum(DEM)]
prob += a1_1 + a1_2 + a1_3 >= DEMtotal * (1/3), "Constraint_1.1"
prob += a2_1 + a2_2 + a2_3 >= DEMtotal * (1/3), "Constraint_1.2"
prob += a3_1 + a3_2 + a3_3 >= DEMtotal * (1/3), "Constraint_1.3"


# monthly production hours at each factory does not exceed 240 hour limit
prob += m1 <= H, "Constraint_2.1"
prob += m2 <= H, "Constraint_2.2"
prob += m3 <= H, "Constraint_2.3"

# production lines can run simultaneously but each line's monthly production hours is no more than factory's monthly production hours
prob += t1_1 <= m1, "Constraint_3.1.1"
prob += t1_2 <= m1, "Constraint_3.1.2"
prob += t1_3 <= m1, "Constraint_3.1.3"

prob += t2_1 <= m2, "Constraint_3.2.1"
prob += t2_2 <= m2, "Constraint_3.2.2"
prob += t2_3 <= m2, "Constraint_3.2.3"

prob += t3_1 <= m3, "Constraint_3.3.1"
prob += t3_2 <= m3, "Constraint_3.3.2"
prob += t3_3 <= m3, "Constraint_3.3.3"

# units produced by each production line equals (line's productivity in units per hour) x (line's monthly production hours)
prob += a1_1 == B1_1 * t1_1, "Constraint_4.1.1"
prob += a1_2 == B1_2 * t1_2, "Constraint_4.1.2"
prob += a1_3 == B1_3 * t1_3, "Constraint_4.1.3"

prob += a2_1 == B2_1 * t2_1, "Constraint_4.2.1"
prob += a2_2 == B2_2 * t2_2, "Constraint_4.2.2"
prob += a2_3 == B2_3 * t2_3, "Constraint_4.2.3"

prob += a3_1 == B3_1 * t3_1, "Constraint_4.3.1"
prob += a3_2 == B3_2 * t3_2, "Constraint_4.3.2"
prob += a3_3 == B3_3 * t3_3, "Constraint_4.3.3"


# number of units factory sends out to retail each month is no more than that factory's monthly production (sum of lines' monthly production)
prob += x1_1 + x1_2 + x1_3 + x1_4 + x1_5 + x1_6 + x1_7 + x1_8 <= a1_1 + a1_2 + a1_3, "Constraint_5.1"
prob += x2_1 + x2_2 + x2_3 + x2_4 + x2_5 + x2_6 + x2_7 + x2_8 <= a2_1 + a2_2 + a2_3, "Constraint_5.2"
prob += x3_1 + x3_2 + x3_3 + x3_4 + x3_5 + x3_6 + x3_7 + x3_8 <= a3_1 + a3_2 + a3_3, "Constraint_5.3"


# each line's monthly production does not exceed (line's productivity in units per hour) x (240 hour limit) 
prob += a1_1 <= B1_1 * H, "Constraint_6.1.1" 
prob += a1_2 <= B1_2 * H, "Constraint_6.1.2"
prob += a1_3 <= B1_3 * H, "Constraint_6.1.3"

prob += a2_1 <= B2_1 * H, "Constraint_6.2.1"   
prob += a2_2 <= B2_2 * H, "Constraint_6.2.2"
prob += a2_3 <= B2_3 * H, "Constraint_6.2.3"

prob += a3_1 <= B3_1 * H, "Constraint_6.3.1"  
prob += a3_2 <= B3_2 * H, "Constraint_6.3.2"
prob += a3_3 <= B3_3 * H, "Constraint_6.3.3"

# total number of units sent out to each store is no less than demand at that store
prob += x1_1 + x2_1 + x3_1 >= DEM1, "Constraint_7.1"
prob += x1_2 + x2_2 + x3_2 >= DEM2, "Constraint_7.2"
prob += x1_3 + x2_3 + x3_3 >= DEM3, "Constraint_7.3"
prob += x1_4 + x2_4 + x3_4 >= DEM4, "Constraint_7.4"
prob += x1_5 + x2_5 + x3_5 >= DEM5, "Constraint_7.5"
prob += x1_6 + x2_6 + x3_6 >= DEM6, "Constraint_7.6"
prob += x1_7 + x2_7 + x3_7 >= DEM7, "Constraint_7.7"
prob += x1_8 + x2_8 + x3_8 >= DEM8, "Constraint_7.8"

# number of units transported is no more than truck capacity
prob += x1_1 <= TC * y1_1, "Constraint_8.1.1"
prob += x1_2 <= TC * y1_2, "Constraint_8.1.2"
prob += x1_3 <= TC * y1_3, "Constraint_8.1.3"
prob += x1_4 <= TC * y1_4, "Constraint_8.1.4"
prob += x1_5 <= TC * y1_5, "Constraint_8.1.5"
prob += x1_6 <= TC * y1_6, "Constraint_8.1.6"
prob += x1_7 <= TC * y1_7, "Constraint_8.1.7"
prob += x1_8 <= TC * y1_8, "Constraint_8.1.8"

prob += x2_1 <= TC * y2_1, "Constraint_8.2.1"
prob += x2_2 <= TC * y2_2, "Constraint_8.2.2"
prob += x2_3 <= TC * y2_3, "Constraint_8.2.3"
prob += x2_4 <= TC * y2_4, "Constraint_8.2.4"
prob += x2_5 <= TC * y2_5, "Constraint_8.2.5"
prob += x2_6 <= TC * y2_6, "Constraint_8.2.6"
prob += x2_7 <= TC * y2_7, "Constraint_8.2.7"
prob += x2_8 <= TC * y2_8, "Constraint_8.2.8"

prob += x3_1 <= TC * y3_1, "Constraint_8.3.1"
prob += x3_2 <= TC * y3_2, "Constraint_8.3.2"
prob += x3_3 <= TC * y3_3, "Constraint_8.3.3"
prob += x3_4 <= TC * y3_4, "Constraint_8.3.4"
prob += x3_5 <= TC * y3_5, "Constraint_8.3.5"
prob += x3_6 <= TC * y3_6, "Constraint_8.3.6"
prob += x3_7 <= TC * y3_7, "Constraint_8.3.7"
prob += x3_8 <= TC * y3_8, "Constraint_8.3.8"

# sum of all factories monthly production (sum of all line's month production) is at least the sum of monthly demand at all retail stores
prob += a1_1 + a1_2 + a1_3 + a2_1 + a2_2 + a2_3 + a3_1 + a3_2 + a3_3 >= DEMtotal, "Constraint_9"
In [ ]:
 
^^^TO OPTIMIZE ANOTHER MONTH RERUN FROM (START)^^^¶


 
![image](https://github.com/user-attachments/assets/bd72cbd6-d66d-46f9-a07c-023db4a71466)
