import random
from pyspark import SparkContext
import time

# Define SparkContext object
sc = SparkContext(appName="EstimatingPi")


# Random x, y points between 0 and 1
def inside(p=None):
    x, y = random.random(), random.random()

    return x * x + y * y < 1


num_samples = int(1e8)

start = time.time()
count = sc.parallelize(range(0, num_samples)).filter(inside).count()
print(f"Pi is approximately {(4.0 * count / num_samples)}")
end = time.time()
print("Runtime: ", end - start)

count = 0
start = time.time()
for i in range(num_samples):
    res = inside()
    count += int(res)
print(f"Pi is approximately {(4.0 * count / num_samples)}")
end = time.time()
print("Runtime: ", end - start)

