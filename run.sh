make all

echo ""
echo "<<<"
mpirun -np 5 ./lab3 9 9 360 0.00025 0.1 5
echo ">>>"

echo ""
echo "<<<"
mpirun -np 5 ./lab3 5 9 300 0.01 1 5
echo ">>>"

make clean
