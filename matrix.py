"""Demonstration of multiplying two matrices using Spark"""
from pyspark import RDD, SparkConf
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import IndexedRowMatrix, IndexedRow, BlockMatrix
from tqdm import tqdm
import numpy as np
import time

N = 10  # Matrix size
MAX_SEQUENTIAL_N = 250


def random_matrix(n: int, m: int, low: int = 0, high: int = 1000) -> np.ndarray:
    """Constructs a matrix of size `n` by `m` with random integer values between `low` and `high`"""
    return np.random.randint(low, high, (n, m))


def empty_matrix(n: int, m: int) -> np.ndarray:
    """Constructs a matrix of size `n` by `m` without initialising values"""
    return np.empty((n, m), dtype=int)


def matrix_multiply(a: np.ndarray, b: np.ndarray, c: np.ndarray, n: int) -> None:
    """Multiplies two square matrices of order `n`, `a` and `b`, and stores result in `c`"""
    for i in tqdm(range(n), disable=(N <= 100)):
        for j in range(n):
            total = 0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total


def as_block_matrix(rdd: RDD) -> BlockMatrix:
    """Converts an `RDD` representing a matrix into a `BlockMatrix`"""
    # Adapted from https://stackoverflow.com/questions/37766213/spark-matrix-multiplication-with-python
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))
    ).toBlockMatrix()


def main() -> None:
    # Initialise Spark
    conf = SparkConf().setMaster(f"local[*]").setAppName("Matrix multiplication")
    spark: SparkSession = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext

    # Generate matrices
    print(f"Matrix size: {N} by {N}")
    a = random_matrix(N, N)
    b = random_matrix(N, N)
    c = empty_matrix(N, N)

    # Sequential matrix multiplication
    if N <= MAX_SEQUENTIAL_N:
        print("Sequential matrix multiplication")
        start = time.time()
        matrix_multiply(a, b, c, N)
        end = time.time()
        print(f"Sequential runtime: {(end - start):.3}s")
    else:
        print(
            "Skipping sequential matrix multiplication, N larger than MAX_SEQUENTIAL_N"
        )

    # Parallel using Spark
    print(f"Parallel matrix multiplication")

    # Construct RDDs
    a_rdd = sc.parallelize(a)
    b_rdd = sc.parallelize(b)
    num_partitions = a_rdd.getNumPartitions()

    # Convert to BlockMatrix used for multiplication
    start = time.time()
    a_bm = as_block_matrix(a_rdd)
    b_bm = as_block_matrix(b_rdd)
    end = time.time()
    time_convert = end - start

    # Multiply
    start = time.time()
    c_bm = a_bm.multiply(b_bm)
    end = time.time()
    time_multiply = end - start

    time_total = time_convert + time_multiply
    print(
        f"Parallel runtime: {time_total:.3}s ({time_convert:.3}s convert, {time_multiply:.3}s multiply, {num_partitions} partitions, {c_bm.numColBlocks} column blocks, {c_bm.numRowBlocks} row blocks used)"
    )

    # Persist the server
    print(
        f"Calculations complete. Spark UI running on {sc.uiWebUrl}. Control-C to quit."
    )
    while True:
        pass


if __name__ == "__main__":
    main()
