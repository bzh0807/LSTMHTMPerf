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

# Prepare textfile and tokenize:
def tokenize(outfile, infile):
  outfile = file(outdir+outfile, "w")
  alnum = re.compile(r"\W+")
  words = defaultdict(int)
  stopwords = []
  count = 0
  with open(infile) as inp:
    #count = 0
    prev_line = ""
    for line in inp:
      #count += 1
      #if count >= 15:
      # break
      for line in line.split(" "):
        if line == "[Illustration]":
          pass
        token = alnum.sub("", line).lower()
        if token not in stopwords:
          outfile.write(token+"\n")
          count += 1

  inp.close()
  outfile.close()
  return count

c1 = tokenize("tokens.txt", "ptb.train.txt")
c2 = tokenize("tokens2.txt", "ptb.test.txt")
c3 = tokenize("tokens3.txt", "ptb.valid.txt")
print c1, c2, c3

#Print total vocabulary words
print len(list(set(map(str.strip, open(outdir+"tokens.txt").readlines()))))

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
          "w": 40, #40
          "n": 2048, #2048
          "forced": True
        }
      },
      "sensorAutoReset" : None,
    },
      "spEnable": True,
      "spParams": {
        "spVerbosity" : 0,
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
        "cellsPerColumn": 32, #32, 16
        "inputWidth": 2048, #2048
        "seed": 1960,
        "temporalImp": "tm_cpp", #"cpp"
        "newSynapseCount": 20,
        "maxSynapsesPerSegment": 32, #32, 8
        "maxSegmentsPerCell": 128, #128, 16
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
      "steps": "1", #"1"
      "maxCategoryCount": 10000,
    },
    "trainSPNetOnlyIfRequested": False,
    "maxPredictionsPerStep": 100000, #50
    "minLikelihoodThreshold": 0, #0.0001
  },
}
#print MODEL_PARAMS

#Helper function for perplexity calculation
def negativeLogLikelihood(dist, bI):
  negLL = 0
  if bI is not None:
    minProb = 1e-12
    for step in dist.keys():
      outOfBucketProb = 1 - sum(dist[step].values())
      if bI in dist[step].keys():
        prob = dist[step][bI]
      else:
        prob = minProb
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
        tmRegion = model._getTPRegion()
        tm = tmRegion.getSelf()._tfdr
        print 'Num Active Cells: ', len(tm.getActiveCells())
  inp.close()

  return np.exp(err/count)
  

model = ModelFactory.create(MODEL_PARAMS)
model.enableInference({"predictedField": "token"})
shifter = InferenceShifter()

#model.load(outdir+"ptbtrainedmodel.model")
#model.enableLearning()

#Log perplexity scores/runtimes
trainperp = {}
validperp = {}
testperp = {}
traintimes = {}
validtimes = {}

NUM_EPOCHS = 2

for n in range(NUM_EPOCHS):
  print 'Training... (',str(n+1),')'
  starttime = time.time()
  train_perp = run_epoch(model, "tokens.txt")
  endtime = time.time()
  trainperp[n] = train_perp
  print 'Training Time:', (endtime-starttime)
  traintimes[n] = endtime-starttime
  model.disableLearning()
  
  print 'Validating...',str(n+1),')'
  starttime = time.time()
  valid_perp = run_epoch(model, "tokens3.txt")
  endtime = time.time()
  validperp[n] = valid_perp
  print 'Valid Time:', (endtime-starttime)
  validtimes[n] = endtime-starttime
  model.enableLearning()

try:
  model.save(outdir+"ptbtrainedmodel")
except:
  print 'Save Error!'

model.disableLearning()
print 'Testing...'
#Prompt for user input so testing can be separately profiled
cont = raw_input("Input Something to Continue!")

starttime = time.time()
test_perp = run_epoch(model, "tokens2.txt")
endtime = time.time()
testperp[0] = test_perp

print 'C/Col: 8, Syn/Seg: 8, Seg/C: 16'
print 'Training'
print trainperp
print 'Validation'
print validperp
print 'Testing'
print testperp
print 'Test Time:', (endtime-starttime)
print 'Training Times'
print traintimes
print 'Valid Times'
print validtimes
