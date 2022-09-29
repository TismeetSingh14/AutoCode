# Auto Code Sequence Generator
The following code is based on the following research paper : [research paper](https://arxiv.org/abs/1705.07962)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage

```python
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays
./train.py ../datasets/web/training_features ../bin

# train with generator to avoid having to fit all the data in memory (RECOMMENDED)
./train.py ../datasets/web/training_features ../bin 1
```
## Generate code for batch of GUIs:

```python
mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./generate.py ../bin pix2code ../gui_screenshots ../code

# equivalent to command above
./generate.py ../bin pix2code ../gui_screenshots ../code greedy

# generate DSL code with beam search and a beam width of size 3
./generate.py ../bin pix2code ../gui_screenshots ../code 3
```

## License
[MIT](https://choosealicense.com/licenses/mit/)