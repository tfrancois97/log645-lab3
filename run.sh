make all

echo ""
echo "<<<"
mpirun -np 7 ./lab3 9 9 360 5 0.1 7
echo ">>>"

echo ""
echo "<<<"
mpirun -np 7 ./lab3 5 9 300 5 1 7
echo ">>>"

make clean
