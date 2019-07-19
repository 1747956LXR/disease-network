kernprof -l model/cHawk.py 
rm efficiency_analysis.txt
python -m line_profiler cHawk.py.lprof >> efficiency_analysis.txt