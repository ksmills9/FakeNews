# Fake News Detection

## Dataset Description

* data.csv: A  dataset with the following attributes:
  * id: unique id for a news article
  * title: the title of a news article
  * author: author of the news article
  * text: the text of the article; could be incomplete
  * label: a label that marks the article as potentially unreliable
    * 1: unreliable
    * 0: reliable

## File Structure
The file structure is the following

```
.
|
+-- datasets
|   +-- data.csv
+-- FakeNewsDetection.py
```

## Try It Out

1. Clone the repo to your local machine-  
`> git clone `  
`> cd directory`

2. Make sure you have all the dependencies installed-  
 * python 3.9.9+
 * pandas
 * matplotlib
 * sklearn
 * textblob
 
3. Run it -  
`> python3 FakeNewsDetection.py`
* Image of Accuracy vs K will displayed during execution, close it after reviewing to allow the program keep executing.

## References
