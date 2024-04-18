# Simulating human robot collaboration

### Setup
1) Initialize venv using `requirements.txt`
2) Run `mat_to_csv.py` to generate csv input files
3) Run `visualization.py` for different visualizations

### Output
#### Visualization
1)  `output/exec_time_plots` contains a subfolder for each operator and histograms for each task representing the frequency of durations of execution times of that specific task (10 bins)
2)  `output/visualize_2d_stress_time` contains 2D histograms for each operator for each task measure frequency of binned task execution lenght and stress levels
3)  `output/visualize_stress_totaltime` plots the average evolution of stress levels wrt the time 