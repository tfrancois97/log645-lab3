#include <chrono>
#include <cstring>
#include <thread>
#include <iostream>
#include <iomanip>

#include <mpi.h>

#include "solver.hpp"
#include "../matrix/matrix.hpp"
#include "../output/output.hpp"

using std::memcpy;
using namespace std::chrono;

using std::cout;
using std::endl;
using std::fixed;
using std::flush;
using std::setprecision;
using std::setw;

using std::this_thread::sleep_for;
using std::chrono::microseconds;
//cd /mnt/d/ETS/E2021/LOG645/labo/Labo3/log645-lab3/

void solveSeq(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix) {
    double c, l, r, t, b;
    
    double h_square = h * h;

    double * linePrevBuffer = new double[cols];
    double * lineCurrBuffer = new double[cols];

    for(int k = 0; k < iterations; k++) {

        memcpy(linePrevBuffer, matrix[0], cols * sizeof(double));
        for(int i = 1; i < rows - 1; i++) {

            memcpy(lineCurrBuffer, matrix[i], cols * sizeof(double));
            for(int j = 1; j < cols - 1; j++) {
                c = lineCurrBuffer[j];
                t = linePrevBuffer[j];
                b = matrix[i + 1][j];
                l = lineCurrBuffer[j - 1];
                r = lineCurrBuffer[j + 1];


                sleep_for(microseconds(sleep));
                matrix[i][j] = c * (1.0 - 4.0 * td / h_square) + (t + b + l + r) * (td / h_square);
            }

            memcpy(linePrevBuffer, lineCurrBuffer, cols * sizeof(double));
        }
    }
}

void solvePar(int rows, int cols, int iterations, double td, double h, int sleep, double ** matrix, int nbProcessors) {

    char hostname[MPI_MAX_PROCESSOR_NAME];
    int rank, procCount, len;
    int tag_send = 0;
    int tag_recv = tag_send;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);
    MPI_Get_processor_name(hostname, &len);
    MPI_Status status;
    
    //double c, t, b, l, r;
    
    //double h_square = h * h;

    int widthProcess = (cols - 2) / nbProcessors;
    int extraProcess = (cols - 2) % nbProcessors;

    if (rank < extraProcess) {
        ++widthProcess;
    }

    double temporaryMatrix[2][widthProcess + 2][rows];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j <= widthProcess + 1; ++j)
        {
            if (i == 0 || i == rows-1 || j == 0 || j == widthProcess + 1){
                temporaryMatrix[0][j][i] = 0;
                temporaryMatrix[1][j][i] = 0;
            }
            else {
                int currentWidthProcess = widthProcess;
                if (rank < extraProcess) {
                    ++currentWidthProcess;
                }.
                int globalCol = j + rank * currentWidthProcess;
                if (rank >= extraProcess) {
                    globalCol += extraProcess;
                }
                temporaryMatrix[0][j][i] = i * (rows - i - 1) * globalCol * (cols - globalCol - 1);
            }
        }
    }

    for (int k = 1; k <= iterations; ++k)
    {
        int currentIndex = k%2;
        int previousIndex = 1 - currentIndex;
        if (nbProcessors > 1 && rank < nbProcessors)
        {
            if (rank == 0)
            {
                // Send right, receive right.
                MPI_Sendrecv(temporaryMatrix[previousIndex][widthProcess], rows, MPI_DOUBLE, rank + 1, tag_send, temporaryMatrix[previousIndex][widthProcess + 1], rows, MPI_DOUBLE, rank + 1, tag_recv, MPI_COMM_WORLD, &status);
            }
            // Send left, receive left.
            else if (rank == nbProcessors - 1)
            {
                MPI_Sendrecv(temporaryMatrix[previousIndex][1], rows, MPI_DOUBLE, rank - 1, tag_send, temporaryMatrix[previousIndex][0], rows, MPI_DOUBLE, rank - 1, tag_recv, MPI_COMM_WORLD, &status);
            }
            else
            {
                // Send right, receive left.
                MPI_Sendrecv(temporaryMatrix[previousIndex][widthProcess], rows, MPI_DOUBLE, rank + 1, 0, temporaryMatrix[previousIndex][0], rows, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &status);
                // Send left, receive right.              
                MPI_Sendrecv(temporaryMatrix[previousIndex][1], rows, MPI_DOUBLE, rank - 1, 0, temporaryMatrix[previousIndex][widthProcess + 1], rows, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &status);
            }
        }

        for (int i = 1; i < rows - 1; ++i) {
            for (int j = 1; j <= widthProcess; ++j)
            {
                sleep_for(microseconds(sleep));
                temporaryMatrix[currentIndex][j][i] = 
                    (1.0 - 4.0 * (td/(h*h))) * temporaryMatrix[previousIndex][j][i] 
                    + (td/(h*h)) * (temporaryMatrix[previousIndex][j][i-1] 
                                    + temporaryMatrix[previousIndex][j][i+1] 
                                    + temporaryMatrix[previousIndex][j-1][i] 
                                    + temporaryMatrix[previousIndex][j+1][i]); 
            }
        }
    }

    double finalMat[cols][rows];
    for (int i = 0; i < rows; ++i) {
        finalMat[0][i] = 0;
        finalMat[cols-1][i] = 0;
    }

    MPI_Datatype subArray;

    int defaultvalue[2] = {1,0};
    int subsizes[2]  = {widthProcess, rows};
    int largerSizes[2]  = {widthProcess + 2, rows};
    MPI_Type_create_subarray(2, largerSizes, subsizes, defaultvalue, MPI_ORDER_C, MPI_DOUBLE, &subArray);
    MPI_Type_commit(&subArray);

    MPI_Gather(&(temporaryMatrix[iterations%2][0][0]), 1, subArray, &(finalMat[1][0]), widthProcess * rows, MPI_DOUBLE, 0, MPI_COMM_WORLD ); 

    if (rank == 0)
    {
        cout << "-----  PARALLEL  -----" << endl << flush;
        // Print matrix
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                cout << fixed << setw(12) << setprecision(2) << finalMat[j][i] << flush;
            }
            cout << endl << flush;
        }
    }
}
