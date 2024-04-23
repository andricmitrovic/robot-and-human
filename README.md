# Simulating human robot collaboration

### Setup
1) Initialize venv using `requirements.txt`
2) Run `mat_to_csv.py` to generate csv input files
3) Run `visualization.py` for different visualizations

### Output
#### Visualization
1)  `output/exec_time_frequency` contains a subfolder for each operator and histograms for each task representing the frequency of durations of execution times of that specific task (10 bins)
2)  `output/exec_time_per_cycle` contains evolution of the task execution time in each next cycle
3)  `output/exec_time_stress_frequency` contains 2D histograms for each operator for each task measure frequency of binned task execution lenght and stress levels
4)  `output/stress_over_time` plots the average evolution of stress levels as total time increases 