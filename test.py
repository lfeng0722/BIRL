import time
from multiprocessing.dummy import Pool as ThreadPool
def process(item):
  print('正在并行for循环')
  print(item)
  time.sleep(5)
items = ['apple', 'bananan', 'cake', 'dumpling']
pool = ThreadPool()
pool.map(process, items)
pool.close()
pool.join()