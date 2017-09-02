# LSTMHTMPerf
## LSTM ##
Run the ptb_REF.py and ptb_BIN.py files using a command like so:
```
python ptb_BIN.py --save_path=SAVEPATH --sample_mode=True --model=MODELSIZE --data_path="/DIR/LSTM HTM Code/simple-examples/data/"
```
* --save_path is the directory where the model is to be stored
* --sample_mode=True turns off training and only performs validation and/or testing
* --model specifies size of model; can be small, medium, or large
* --data_path specifies where the dataset is located

The wiki_REF.py and wiki_BIN.py files are run similarly, except we change the --data_path argument:
```
--data_path="/DIR/LSTM HTM Code/wiki8_data/"
```

## HTM ##
Run the runptbbatchload.py and runwiki.py files without special arguments.
```
python runptbbatchload.py
```
```
python runwiki.py
```





