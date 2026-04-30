using Plots, Dates

labels = ["Finite Diff", "BBO", "ForwardDiff"]
times = [Time(1, 31, 11), Time(0, 00, 51), Time(0, 02, 58)]

seconds = [Dates.value(t) / 1000000000 for t in times]
time_labels = Dates.format.(times, "MM:SS")

bar(labels, seconds, label=false, ylabel="Time (seconds)", title="Training Time over 2000 Epochs")
