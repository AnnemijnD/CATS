import pickle
with open('results_heatmap.pkl', 'rb') as f1:
    prev_results = pickle.load(f1)

with open('par_opt.pkl', 'rb') as f2:
    par_opt = pickle.load(f2)
# print(prev_results)
print(par_opt)
