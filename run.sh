# Load args
echo "Running Instant-NGP with $1 and $2"

./Instant-NGP $1 800 $2
cp ./near.txt ~/Workspace/PCAccNR-Scheduler/ 
cp ./far.txt ~/Workspace/PCAccNR-Scheduler/