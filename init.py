import lzma
import tarfile
import os
import subprocess
import shutil

from contextlib import closing

SAPRSE_GRAPH_TO_GEN = [[5000, 0.001], [10000, 0.001], [10000, 0.01], [10000, 0.1], [20000, 0.001], [40000, 0.001], [80000, 0.001]]
FULL_GRAPH_TO_GEN = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000, 50000, 80000]
NUMBER_OF_WORST_CASES = 12

DATA_ROOT_DIR = './data/graphs/'
MATRICES_DIR = 'Matrices'
DATA_TO_UNPACK = ['MemoryAliases', 'RDF']
GT_GRAPH = './GTgraph/random/GTgraph-random'
TMP_FILE = 'tmp.txt'


def unpack(file_from, path_to):
    with lzma.open(file_from) as f:
        with tarfile.open(fileobj=f) as tar:
            content = tar.extractall(path_to)

def unpack_graphs():
    for d in DATA_TO_UNPACK:
        to = os.path.join(DATA_ROOT_DIR, d)
        arch = os.path.join(to, '%s.tar.xz'%(MATRICES_DIR))
        print ('Unpack ', arch, ' to ', to)
        unpack(arch, to)

def gen_sparse_graph(target_dir, vertices, prob):
 
    subprocess.run([GT_GRAPH, '-t', '0', '-n', '%s'%(vertices), '-p', '%s'%(prob), '-o', TMP_FILE])
    
    with open(os.path.join(target_dir, 'G%sk-%s.txt'%(int(vertices/1000), prob)), 'a') as out_file:
        with open(TMP_FILE) as in_file:
            for l in in_file.readlines():
                if l.startswith('a '):
                   a = l.split(' ')
                   out_file.write('%s A %s \n'%(a[1], a[2]))
                   out_file.write('%s AR %s \n'%(a[2], a[1]))
    os.remove(TMP_FILE)

def gen_worst_case_graph(target_dir, vertices):
    first_cycle = int(vertices / 2) + 1
    second_cycle = int(vertices / 2) - 1
    
    with open(os.path.join(target_dir, 'worstcase_%s.txt'%(vertices)), 'a') as out_file:
        for i in range(0, first_cycle - 1):
            out_file.write('%s A %s \n'%(i, i + 1))
        out_file.write('%s A %s \n'%(first_cycle - 1, 0))
        
        out_file.write('%s B %s \n'%(first_cycle - 1, first_cycle))      

        for i in range(first_cycle, vertices - 1):
            out_file.write('%s B %s \n'%(i, i + 1))
        out_file.write('%s B %s \n'%(vertices - 1, first_cycle - 1))

def gen_cycle_graph(target_dir, vertices):
    with open(os.path.join(target_dir, 'fullgraph_%s.txt'%(vertices)), 'a') as out_file:
        for i in range(0, vertices - 1):
            out_file.write('%s A %s \n'%(i, i + 1))
        out_file.write('%s A %s \n'%(vertices - 1, 0))   

def clean_dir(path):
   if os.path.isdir(path): 
      shutil.rmtree(path)
   os.mkdir(path)

def generate_all_sparse_graphs():
   print('Sparse graphs generation is started.') 
   matrices_dir = os.path.join(os.path.join(DATA_ROOT_DIR, 'SG'), MATRICES_DIR)
   clean_dir(matrices_dir)

   for g in SAPRSE_GRAPH_TO_GEN: gen_sparse_graph(matrices_dir, g[0], g[1])
   print('Sparse graphs generation is finished.')


def generate_full_graphs():
   print('Full graphs generation is started.')
   matrices_dir = os.path.join(os.path.join(DATA_ROOT_DIR, 'FullGraph'), MATRICES_DIR)
   clean_dir(matrices_dir)
   
   for g in FULL_GRAPH_TO_GEN: gen_cycle_graph(matrices_dir, g)
   print('Full graphs generation is finished.')

def generate_worst_case_graphs():
   print('Worst case graphs generation is started.')
   matrices_dir = os.path.join(os.path.join(DATA_ROOT_DIR, 'WorstCase'), MATRICES_DIR)
   clean_dir(matrices_dir)
   
   for n in range(2, NUMBER_OF_WORST_CASES): gen_worst_case_graph(matrices_dir, 2 ** n)
   print('Worst case graphs generation is finished.')


if __name__ == '__main__':
   unpack_graphs()
   generate_all_sparse_graphs()
   generate_full_graphs()
   generate_worst_case_graphs()
