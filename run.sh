make all

echo ""
echo "<<<"
mpirun -np 8 ./lab3 9 10 360 5 0.1 8
echo ">>>"

echo ""
echo "<<<"
mpirun -np 8 ./lab3 5 10 300 5 1 8
echo ">>>"

make clean
