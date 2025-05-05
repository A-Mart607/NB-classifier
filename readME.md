packages needed:
- OS
- Json
- sys

Small NB Example is in the /small-NB folder

# How to run

```bash 
# get the train vectors
python .\pre-process.py .\test
python .\pre-process.py .\train

# run the model
python NB.py .\train.vectors .\test.vectors .\movie-review.NB .\report.txt

```