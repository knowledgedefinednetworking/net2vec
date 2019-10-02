# Copyright (c) 2018-2019, Krzysztof Rusek
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# author: Krzysztof Rusek, AGH

import sys
import os
import pandas as pd
import tarfile
import re
import io
import itertools as it
import glob
import tensorflow as tf
from multiprocessing import Pool
import argparse
from random import shuffle
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def ned2lists(fname):
    '''
    Args:
        fname a textFile path
    Returns:
        connections - list of lists neighbors for each node
        n - number of nodes
        edges - a list of pars (srd,dst)
        capacities - link capcities
    '''
    channels = []
    capacities = []

    p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps\s+<-->\s+node(\d+).port\[(\d+)\]')
    with open(fname) as fobj:
        for line in fobj:
            m=p.match(line)
            if m:
                matches = list(map(int,m.groups()))
                capacities.append(matches[2])
                del matches[2]
                channels.append(matches)
    n=max(map(max, channels))+1
    
    connections = [{} for i in range(n)]
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    connections = [[v for k,v in sorted(con.items())] 
                   for con in connections ]
    edges = [(c[0],c[2]) for c in channels] + [(c[2],c[0]) for c in channels]
    capacities = capacities + capacities
    return connections,n,edges,capacities


def load_routing(routing_file):
    '''
    Loads routing descriptions
    Args:
        routing_file: a file handle
    '''
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values


def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def make_indices(paths):
    link_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return link_indices, path_indices, sequ_indices


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class UPCDataset:
    '''
    Helper converter of UPC datasets format to tfrecords
    Supports UPC dataset v2 (zip of tar.gz files)
    '''
        
    def __init__(self, ned_file, output_dir):
        (self.connections,
        self.n,
        self.edges,
        self.capacities ) = ned2lists(ned_file)
        self.output_dir = output_dir
        
        
    def __call__(self, tarpath):
        '''
        Reads selected dataset from a zipped tar file and save it as tfrecord file
        '''
        logging.info('Started %s...',tarpath)
        tarname = os.path.split(tarpath)[1]
        routing_name = tarname.replace('.tar.gz','/Routing.txt')
        data_name = tarname.replace('.tar.gz','/simulationResults.txt')
        tfrecords_name = os.path.join(self.output_dir,tarname.replace('tar.gz','tfrecords'))
        
        with tarfile.open(tarpath) as tar:
            with tar.extractfile(routing_name) as fobj:
                R = load_routing(fobj)
            with tar.extractfile(data_name) as fobj:
                names, drop_names = self._make_names()
                data = pd.read_csv(fobj,header=None, names=names,index_col=False)
                data = data.drop(drop_names, axis=1)
        
        paths = self._make_paths(R)
        link_indices, path_indices, sequ_indices = make_indices(paths)
        
        delay = data.filter(regex='average_delay').values
        logdelay = data.filter(regex='average_log').values
        tm = data.filter(regex='traffic').values
        jitter = data.filter(regex='variance').values
        drops =  data.filter(regex='drops').values
        packets =  data.filter(regex='packets').values

        n_paths = delay.shape[1]
        n_links = max(max(paths))+1
        n_total = len(path_indices)

        writer = tf.python_io.TFRecordWriter(tfrecords_name)
        
        for item in zip(tm,delay,jitter,drops,packets,logdelay):
            example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(item[0]),
            'delay':_float_features(item[1]),
            'jitter':_float_features(item[2]),
            'drops':_float_features(item[3]),
            'packets':_float_features(item[4]),
            'logdelay':_float_features(item[5]),
            'links':_int64_features(link_indices),
            'paths':_int64_features(path_indices),
            'sequences':_int64_features(sequ_indices),
            'n_links':_int64_feature(n_links), 
            'n_paths':_int64_feature(n_paths),
            'n_total':_int64_feature(n_total),
            'capacities':_float_features(self.capacities)

            }))
            
            writer.write(example.SerializeToString())
        writer.close()
        logging.info('Finished %s...',tarpath)

        
    def _make_paths(self, R):
        '''
        Construct path description for a given omnet++ routing
        Args:
            R - Routing matrix in omnet++ format
        Returns:
            A list of link indices for each path
        '''
        paths=[]
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    paths.append([self.edges.index(tup) for tup in pairwise(genPath(R,i,j,self.connections))])
        return paths

    def _make_names(self):
        '''
        Generate names for dataset columns
        '''
        n=self.n
        names=[]

        counter=0
        drop_names=[]

        for i in range(n):
            for j in range(n):
                names.append('traffic_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
                names.append('packets_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
                names.append('drops_{}_{}'.format(i,j))
                if i == j: drop_names.append(names[-1])
                counter += 1
        for i in range(n):
            for j in range(n):
                for k in ['average','average_log' ,'q10','q20','q50','q80','q90','variance']:
                    names.append('{}_delay_{}_{}'.format(k,i,j))
                    if i == j: drop_names.append(names[-1])
                    counter += 1
        names.append('empty')
        return names, drop_names
                        
                    
DESCRIPTION='''
UPC dataset processor
Helper command to unzip files:
 find . -name '*.zip' | xargs -P 4 -L 1 unzip -u
'''
def main():
    '''
    
    
    '''
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('-d', help='Data directory',  type=str ,  required=True, nargs='+')
    parser.add_argument('--processes', help='Number of concurrent jobs', default=2, type=int)
    args = parser.parse_args()

    with Pool(processes=args.processes) as pool:
        for d in args.d:
            for o in ['train','evaluate']:
                os.makedirs( os.path.join(d, o) , exist_ok=True)
        
            ned_file = glob.glob(os.path.join(d, '*.ned'))[0]
            tars = glob.glob(os.path.join(d, '*.tar.gz'))
            shuffle(tars)

            first_n = int(0.7*len(tars))
            processor = UPCDataset(ned_file, os.path.join(d, 'train'))
            pool.map(processor, tars[0:first_n] )

            processor.output_dir = os.path.join(d, 'evaluate')
            pool.map(processor, tars[first_n:] )


if __name__ == '__main__':
    main()
