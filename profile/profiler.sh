kernprof -l cHawk.py 
rm result.txt
python -m line_profiler cHawk.py.lprof >> result.txt