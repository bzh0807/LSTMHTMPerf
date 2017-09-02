# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import csv
from operator import itemgetter
import sys
from collections import defaultdict
import re
import numpy as np
import time

from nupic.data.inference_shifter import InferenceShifter
from nupic.frameworks.opf.model_factory import ModelFactory
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.prediction_metrics_manager import MetricsManager

outdir = "/localtmp/wtz5pp/"
indir = "/af5/wtz5pp/temporary/"

# Prepare textfile and tokenize:
outfile = file(outdir+"wikitrain.txt", "w")
outfile2 = file(outdir+"wikivalid.txt", "w")
#outfile3 = file(outdir+"tokens3.txt", "w")
alnum = re.compile(r"\W+")
words = defaultdict(int)
stopwords = []
"""with open("stopwords.txt") as skip:
  for line in skip:
    line = line[:-1]
    stopwords.append(line)
skip.close()"""
with open(indir+"train2") as inp:
  count = 0
  prev_line = ""
  for line in inp:
    count += 1
    #if count >= 15:
    #  break
    for line in line.split(" "):
      if line == "[Illustration]":
        pass
      token = alnum.sub("", line).lower()
      if token not in stopwords:
        outfile.write(token + "\n")
        count+=1
inp.close()
outfile.close()
print 'Train Size:', count

count = 0
with open(indir+"valid2") as inp:
  count = 0
  prev_line = ""
  for line in inp:
    #count += 1
    #if count >= 15:
    #  break
    for line in line.split(" "):
      if line == "[Illustration]":
        pass
      token = alnum.sub("", line).lower()
      if token not in stopwords:
        outfile2.write(token + "\n")
        count += 1
inp.close()
outfile2.close()
print 'Valid Size:', count

print len(list(set(map(str.strip, open(outdir+"wikitrain.txt").readlines()))))

# Create and run the model:

MODEL_PARAMS = {
  "model": "HTMPrediction",
  "version": 1,
  "predictAheadTime": None,
  "modelParams": {
    "inferenceType": "TemporalMultiStep",
    "sensorParams": {
      "verbosity" : 0,
      "encoders": {
        "token": {
          "fieldname": "token",
          "name": "token",
          "type": "SDRCategoryEncoder",
          "categoryList": list(set(map(str.strip, open(outdir+"tokens.txt").readlines()))),
          "w": 40,
          "n": 2048,
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": True,
      "spParams": {
        "spVerbosity" : 1,
        "globalInhibition": 1,
        "columnCount": 2048, #2048
        "inputWidth": 0,
        "numActiveColumnsPerInhArea": 40,
        "seed": 1956,
        "columnDimensions": 0.5,
        "synPermConnected": 0.1,
        "synPermActiveInc": 0.1,
        "synPermInactiveDec": 0.01,
        "spatialImp": "cpp", #
    },

    "tmEnable" : True,
    "tmParams": {
      "verbosity": 0,
        "columnCount": 2048, #2048
        "cellsPerColumn": 32,
        "inputWidth": 2048, #2048
        "seed": 1960,
        "temporalImp": "tm_cpp", #"cpp"#
        "newSynapseCount": 20,
        "maxSynapsesPerSegment": 32,
        "maxSegmentsPerCell": 128,
        "initialPerm": 0.21,
        "permanenceInc": 0.1,
        "permanenceDec" : 0.1,
        "globalDecay": 0.0,
        "maxAge": 0,
        "minThreshold": 12,
        "activationThreshold": 16,
        "outputType": "normal",
        "pamLength": 1,
      },
      "clParams": {
        "implementation": "cpp", #"py"
        "regionName" : "SDRClassifierRegion",
        "verbosity" : 0,
        "alpha": 0.001,
        "steps": "50",
        "maxCategoryCount": 10000,
      },
      "trainSPNetOnlyIfRequested": False,
      "maxPredictionsPerStep": 100000, #50
      "minLikelihoodThreshold": 0, #0.0001
    },
}

def negativeLogLikelihood(dist, bI):
  negLL = 0
  if bI is not None:
    minProb = 1e-12#0.0000001 #(9), 19: 0.000001
    for step in dist.keys():
      outOfBucketProb = 1 - sum(dist[step].values())
      if bI in dist[step].keys():
        prob = dist[step][bI]
        #print len(dist[step].keys()), 'a'
      else:
        prob = minProb
        #print len(dist[step].keys()), 'b'
      if prob < minProb:
        prob = minProb
      negLL -= np.log(prob)
  return negLL
    
def run_epoch(model, filename):
  err = 0
  dist = {}
  with open(outdir+filename) as inp:
    count = 0
    for line in inp:
      count+=1      
      if count%1000 == 0:
        print 'Step ',count,': ',line
      
      token = line.strip()
      modelInput = {"token": token}
      result = model.run(modelInput)

      bI = result.classifierInput.bucketIndex 
      if dist is not None:
        neglog = negativeLogLikelihood(dist, bI)
      dist = result.inferences["multiStepBucketLikelihoods"]
      if neglog is not None:
        err += neglog
      #if count <= 100:
      #  print 'Run ',count
      if count%250 == 0:
        print 'Perp:',np.exp(err/count)    
  inp.close()

  return np.exp(err/count)
  

model = ModelFactory.create(MODEL_PARAMS)
model.enableInference({"predictedField": "token"})
shifter = InferenceShifter()
#out = csv.writer(open(outdir+"results.csv", "wb"))
#out2 = csv.writer(open(outdir+"results2.csv", "wb"))
#model.load(outdir+"ptbtrainedmodel.model")
#model.enableLearning()
trainperp = {}
validperp = {}
testperp = {}
traindur = 0

for n in range(1):
  print 'Training... (',str(n+1),')'
  st = time.time()
  train_perp = run_epoch(model, "wikitrain.txt")
  et = time.time()
  trainperp[n] = train_perp
  model.disableLearning()
  print 'Training Time:', (et-st)
  traindur = (et-st)
  
  #print 'Validating...',str(n+1),')'
  #valid_perp = run_epoch(model, "wikivalid.txt")
  #validperp[n] = valid_perp
  #model.enableLearning()

model.disableLearning()
print 'Testing...'

starttime = time.time()
test_perp = run_epoch(model, "wikivalid.txt")
endtime = time.time()
testperp[0] = test_perp

print 'Training'
print trainperp
print 'Validation'
print validperp
print 'Testing'
print testperp
print 'Train Time:', traindur
print 'Test Time:', (endtime-starttime)

#try:
#  model.save(outdir+"ptbtrainedmodel.model")
#except:
#  print 'Save Error!'
