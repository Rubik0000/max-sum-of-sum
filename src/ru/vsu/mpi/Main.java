package ru.vsu.mpi;

import mpi.Cartcomm;
import mpi.MPI;
import mpi.ShiftParms;

import java.util.Arrays;
import java.util.Random;

/**
 * Задача 6, пункт 2
 * Даны две матрицы размером n * n (n - количество процессов), матрицы распределены между процессами
 * Каждый процесс генерирует строку A и строку (столбец) B
 * Для вычисления по формуле процесс использует свою строку A, а строки (столбцы) B передаются
 * последовательно каждому процессу, при этом используется топология кольцо
 * res(i) = max(j) sum(k) ( a(i,k) + b(k,j) )
 *
 * Запуск: mpjrun.sh -np 3 ru.vsu.mpi.Main
 */
public class Main {

    public static final int MASTER = 0;


    public static void main(String[] args) {
        try {
            MPI.Init(args);
            int rank = MPI.COMM_WORLD.Rank();
            int size = MPI.COMM_WORLD.Size();
            // формирование строки матрицы A и столбца B
            int[] aStr = randVector(size);
            int[] bCol = randVector(size);

            formAndPrintMatrix(aStr, bCol, rank, size);

            // создание топологии кольцо
            int[] dims = new int[] {size};
            boolean[] periods = new boolean[] {true};
            Cartcomm cartcomm = MPI.COMM_WORLD.Create_cart(dims, periods, false);
            int[] coords = cartcomm.Coords(rank);
            ShiftParms shift = cartcomm.Shift(0, -1);
            int max = Integer.MIN_VALUE;
            // для каждого процесса посчитать сумму, обновить максимум и обменять столбец B
            for (int k = 0; k < size; ++k) {
                int sum = 0;
                for (int j = 0; j < size; ++j) {
                    sum += aStr[j] + bCol[j];
                }
                if (sum > max) {
                    max = sum;
                }
                cartcomm.Sendrecv_replace(bCol, 0, bCol.length, MPI.INT, shift.rank_dest, 0, shift.rank_source, 0);
            }
            System.out.println("Process " + Arrays.toString(coords) + ": " + max);
        } finally {
            MPI.Finalize();
        }
    }

    private static int[] randVector(int size) {
        Random random = new Random();
        int[] vector = new int[size];
        for (int i = 0; i < size; ++i) {
            vector[i] = (random.nextInt(10));
        }
        return vector;
    }

    private static void formAndPrintMatrix(int[] aStr, int[] bCol, int rank, int size) {
        int[] aRecvBuf = null;
        int[] bRecvBuf = null;
        if (rank == MASTER) {
            aRecvBuf = new int[size * size + size];
            bRecvBuf = new int[size * size + size];
        }
        int[] aSendBuf = formSendBuf(aStr, rank);
        int[] bSendBuf = formSendBuf(bCol, rank);
        int[] sendCount = new int[size];
        Arrays.fill(sendCount, size + 1);
        MPI.COMM_WORLD.Gatherv(
                aSendBuf, 0, size + 1, MPI.INT,
                aRecvBuf, 0, sendCount, createBlocksFromSendCount(sendCount), MPI.INT, MASTER
        );
        MPI.COMM_WORLD.Gatherv(
                bSendBuf, 0, size + 1, MPI.INT,
                bRecvBuf, 0, sendCount, createBlocksFromSendCount(sendCount), MPI.INT, MASTER
        );
        if (rank == MASTER) {
            System.out.println("Matrix A");
            printRows(aRecvBuf, size);
            System.out.println("Matrix B");
            printCols(bRecvBuf, size);
        }
    }

    private static void printRows(int[] buf, int size) {
        for (int i = 0; i < size; ++i) {
            int offset = i * (size + 1);
            StringBuilder stringBuilder = new StringBuilder()
                    .append(buf[offset])
                    .append(" [");
            for (int j = 1; j < (size + 1); ++j) {
                stringBuilder
                        .append(buf[offset + j]);
                if (j < size) {
                    stringBuilder.append(", ");
                }
            }
            stringBuilder.append("]");
            System.out.println(stringBuilder.toString());
        }
    }

    private static void printCols(int[] buf, int size) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < size; ++i) {
            stringBuilder.append(" ")
                    .append(buf[i * (size + 1)])
                    .append(" ");
        }
        System.out.println(stringBuilder);
        for (int i = 1; i < (size + 1); ++i) {
            stringBuilder = new StringBuilder();
            stringBuilder.append("[");
            for (int j = 0; j < size; ++j) {
                stringBuilder.append(buf[(size + 1) * j + i]);
                if (j < size - 1) {
                    stringBuilder.append(", ");
                }
            }
            stringBuilder.append("]");
            System.out.println(stringBuilder);
        }
    }

    private static int[] createBlocksFromSendCount(int[] sendCount) {
        int[] displ = new int[sendCount.length];
        int offset = 0;
        for (int i = 0; i < displ.length; ++i) {
            displ[i] = offset;
            offset += sendCount[i];
        }
        return displ;
    }

    private static int[] formSendBuf(int[] buf, int rank) {
        int[] sendBuf = new int[buf.length + 1];
        sendBuf[0] = rank;
        for (int i = 0; i < buf.length; ++i) {
            sendBuf[i + 1] = buf[i];
        }
        return sendBuf;
    }
}
