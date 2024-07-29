# Week 3 Project: Active learning on fashion classifier

```
pip install -e .
```

This project will investigate various strategies to identify elements for relabeling.


## Results

MODEL (DEFAULT).    : 30.2%
MODEL (RANDOM)      : 40.9%
MODEL (UNCERTAINTY) : 38.2%
MODEL (MARGIN)      : 41.5%
MODEL (ENTROPY)     : 39.7%
MODEL (AUGMENT)     : 50.3%


## raw notes
default {'acc': 0.3021000027656555, 'loss': 7.151782989501953}
random {'acc': 0.4092000126838684, 'loss': 1.679159164428711}
uncertainty {'acc': 0.3824000358581543, 'loss': 1.7412782907485962}
margin {'acc': 0.41520002484321594, 'loss': 1.6704493761062622}
entropy {'acc': 0.3968999981880188, 'loss': 3.6368236541748047}
augment on test(?) {'acc': 0.7898362874984741, 'loss': 0.5932655334472656}
augment {'acc': 0.5034000277519226, 'loss': 2.168470621109009}

