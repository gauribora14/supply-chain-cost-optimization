# 🚚 Supply Chain Cost Optimization Using Python

This project applies a Mixed-Integer Linear Programming (MILP) model to optimize production and transportation costs across a multi-plant, multi-retail studio network. The goal is to fulfill demand at the lowest possible cost while respecting plant capacities, logistics constraints, and energy efficiency.

Developed as part of the MSBA 204 course at California State University, Sacramento.

---

## Objective

Minimize total production and transportation costs over a 12-month period by determining:
- How many vehicles each of the three plants (Troy, Newark, Harrisburg) should produce monthly
- How those vehicles should be allocated to eight retail studios

The model considers production line capacity, unit costs, energy usage, emissions, and demand requirements.

---

## Tools & Libraries

- **Python** (core language)
- `pandas` (data processing)
- `PuLP` (MILP optimization)
- `Plotly` (interactive data visualization)
- `numpy`, `openpyxl` (supporting packages)

---

## Project Structure

```text
supply-chain-optimization-msba/
│
├── supply_chain_model.ipynb   # Main notebook with modeling and visualization
├── requirements.txt           # Python dependencies
├── data/
│   └── monthly_demand.xlsx    # Input data (anonymized or simulated)
├── visuals/
│   ├── sankey_flows.png       # Example: Plant-to-studio flow diagram
│   ├── cost_summary_chart.png # Example: Annual cost comparison
│   └── heatmap_utilization.png
└── README.md
